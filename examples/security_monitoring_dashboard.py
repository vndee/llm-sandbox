"""Security monitoring dashboard for LLM Sandbox.

This module provides a comprehensive monitoring system for security policies,
including real-time alerts, metrics collection, and violation tracking.
"""

import json
import time
import threading
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from pathlib import Path

from llm_sandbox import SandboxSession
from llm_sandbox.security import (
    SecurityIssueSeverity,
    SecurityPattern,
    DangerousModule,
    SecurityPolicy,
)


@dataclass
class SecurityEvent:
    """Represents a security event."""
    timestamp: str
    event_type: str  # 'violation', 'execution', 'policy_update'
    severity: str
    user_id: str
    session_id: str
    details: Dict[str, Any]
    code_hash: str
    patterns_matched: List[str]
    modules_blocked: List[str]


@dataclass
class SecurityMetrics:
    """Security metrics for monitoring."""
    total_executions: int
    blocked_executions: int
    violation_rate: float
    top_violations: List[Dict[str, Any]]
    severity_distribution: Dict[str, int]
    hourly_violations: List[int]
    user_violation_counts: Dict[str, int]


class SecurityDatabase:
    """Database for storing security events and metrics."""
    
    def __init__(self, db_path: str = "security_monitoring.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the security database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    details TEXT,
                    code_hash TEXT,
                    patterns_matched TEXT,
                    modules_blocked TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metrics_data TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON security_events(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_type ON security_events(event_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON security_events(user_id)
            """)
    
    def store_event(self, event: SecurityEvent):
        """Store a security event."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO security_events 
                (timestamp, event_type, severity, user_id, session_id, 
                 details, code_hash, patterns_matched, modules_blocked)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.timestamp,
                event.event_type,
                event.severity,
                event.user_id,
                event.session_id,
                json.dumps(event.details),
                event.code_hash,
                json.dumps(event.patterns_matched),
                json.dumps(event.modules_blocked)
            ))
    
    def get_events(
        self, 
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[SecurityEvent]:
        """Retrieve security events with filters."""
        query = "SELECT * FROM security_events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            events = []
            
            for row in cursor.fetchall():
                events.append(SecurityEvent(
                    timestamp=row[1],
                    event_type=row[2],
                    severity=row[3],
                    user_id=row[4],
                    session_id=row[5],
                    details=json.loads(row[6]),
                    code_hash=row[7],
                    patterns_matched=json.loads(row[8]),
                    modules_blocked=json.loads(row[9])
                ))
            
            return events
    
    def store_metrics(self, metrics: SecurityMetrics):
        """Store security metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO security_metrics (timestamp, metrics_data)
                VALUES (?, ?)
            """, (
                datetime.now().isoformat(),
                json.dumps(asdict(metrics))
            ))


class SecurityMonitor:
    """Real-time security monitoring system."""
    
    def __init__(self, db_path: str = "security_monitoring.db"):
        self.db = SecurityDatabase(db_path)
        self.active_sessions = {}
        self.alert_thresholds = {
            'violation_rate_per_hour': 10,
            'high_severity_violations_per_hour': 5,
            'suspicious_user_violations': 3
        }
        self.recent_events = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("🔍 Security monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("🛑 Security monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Calculate and store metrics every minute
                metrics = self.calculate_current_metrics()
                self.db.store_metrics(metrics)
                
                # Check for alerts
                self.check_alerts(metrics)
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(5)
    
    def log_security_event(
        self,
        event_type: str,
        severity: SecurityIssueSeverity,
        user_id: str,
        session_id: str,
        details: Dict[str, Any],
        code_hash: str = None,
        patterns_matched: List[str] = None,
        modules_blocked: List[str] = None
    ):
        """Log a security event."""
        event = SecurityEvent(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            severity=severity.name,
            user_id=user_id,
            session_id=session_id,
            details=details,
            code_hash=code_hash or "",
            patterns_matched=patterns_matched or [],
            modules_blocked=modules_blocked or []
        )
        
        self.db.store_event(event)
        self.recent_events.append(event)
        
        # Immediate alert for high severity events
        if severity == SecurityIssueSeverity.HIGH:
            self.trigger_alert("HIGH_SEVERITY_VIOLATION", event)
    
    def calculate_current_metrics(self) -> SecurityMetrics:
        """Calculate current security metrics."""
        # Get events from last 24 hours
        start_time = (datetime.now() - timedelta(hours=24)).isoformat()
        events = self.db.get_events(start_time=start_time)
        
        total_executions = len([e for e in events if e.event_type == 'execution'])
        violations = [e for e in events if e.event_type == 'violation']
        blocked_executions = len(violations)
        
        violation_rate = (blocked_executions / total_executions) if total_executions > 0 else 0
        
        # Count violations by pattern
        pattern_counts = defaultdict(int)
        for event in violations:
            for pattern in event.patterns_matched:
                pattern_counts[pattern] += 1
        
        top_violations = [
            {'pattern': pattern, 'count': count}
            for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Severity distribution
        severity_dist = defaultdict(int)
        for event in violations:
            severity_dist[event.severity] += 1
        
        # Hourly violations for last 24 hours
        hourly_violations = [0] * 24
        for event in violations:
            event_time = datetime.fromisoformat(event.timestamp)
            hour_diff = int((datetime.now() - event_time).total_seconds() / 3600)
            if 0 <= hour_diff < 24:
                hourly_violations[23 - hour_diff] += 1
        
        # User violation counts
        user_violations = defaultdict(int)
        for event in violations:
            if event.user_id:
                user_violations[event.user_id] += 1
        
        return SecurityMetrics(
            total_executions=total_executions,
            blocked_executions=blocked_executions,
            violation_rate=violation_rate,
            top_violations=top_violations,
            severity_distribution=dict(severity_dist),
            hourly_violations=hourly_violations,
            user_violation_counts=dict(user_violations)
        )
    
    def check_alerts(self, metrics: SecurityMetrics):
        """Check for alert conditions."""
        # Check violation rate
        recent_violations = sum(metrics.hourly_violations[-1:])  # Last hour
        if recent_violations > self.alert_thresholds['violation_rate_per_hour']:
            self.trigger_alert("HIGH_VIOLATION_RATE", {
                'violations_per_hour': recent_violations,
                'threshold': self.alert_thresholds['violation_rate_per_hour']
            })
        
        # Check high severity violations
        high_severity_count = metrics.severity_distribution.get('HIGH', 0)
        if high_severity_count > self.alert_thresholds['high_severity_violations_per_hour']:
            self.trigger_alert("HIGH_SEVERITY_VIOLATIONS", {
                'high_severity_count': high_severity_count,
                'threshold': self.alert_thresholds['high_severity_violations_per_hour']
            })
        
        # Check suspicious users
        for user_id, violation_count in metrics.user_violation_counts.items():
            if violation_count > self.alert_thresholds['suspicious_user_violations']:
                self.trigger_alert("SUSPICIOUS_USER", {
                    'user_id': user_id,
                    'violation_count': violation_count,
                    'threshold': self.alert_thresholds['suspicious_user_violations']
                })
    
    def trigger_alert(self, alert_type: str, data: Any):
        """Trigger a security alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'data': data
        }
        
        print(f"🚨 SECURITY ALERT: {alert_type}")
        print(f"   Data: {data}")
        
        # In a real implementation, you would send this to:
        # - Email notifications
        # - Slack/Teams webhooks
        # - PagerDuty/OpsGenie
        # - Log aggregation systems
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for security dashboard."""
        metrics = self.calculate_current_metrics()
        recent_events = list(self.recent_events)[-50:]  # Last 50 events
        
        return {
            'metrics': asdict(metrics),
            'recent_events': [asdict(event) for event in recent_events],
            'alert_status': {
                'monitoring_active': self.monitoring_active,
                'thresholds': self.alert_thresholds
            },
            'system_health': {
                'database_size': Path(self.db.db_path).stat().st_size if Path(self.db.db_path).exists() else 0,
                'active_sessions': len(self.active_sessions),
                'uptime': time.time()  # In a real system, track actual uptime
            }
        }


class SecureExecutionWrapper:
    """Wrapper for sandbox execution with monitoring integration."""
    
    def __init__(self, security_policy: SecurityPolicy, monitor: SecurityMonitor):
        self.security_policy = security_policy
        self.monitor = monitor
    
    def execute_with_monitoring(
        self,
        code: str,
        user_id: str,
        session_id: str,
        libraries: List[str] = None
    ) -> Dict[str, Any]:
        """Execute code with comprehensive monitoring."""
        code_hash = str(hash(code))
        
        try:
            with SandboxSession(
                lang="python",
                security_policy=self.security_policy,
                verbose=False
            ) as session:
                # Security check
                is_safe, violations = session.is_safe(code)
                
                patterns_matched = [v.pattern for v in violations]
                modules_blocked = []  # Could extract from policy
                
                if not is_safe:
                    # Log security violation
                    self.monitor.log_security_event(
                        event_type="violation",
                        severity=max([v.severity for v in violations], default=SecurityIssueSeverity.LOW),
                        user_id=user_id,
                        session_id=session_id,
                        details={
                            'code_length': len(code),
                            'violation_count': len(violations),
                            'violation_details': [
                                {
                                    'pattern': v.pattern,
                                    'description': v.description,
                                    'severity': v.severity.name
                                } for v in violations
                            ]
                        },
                        code_hash=code_hash,
                        patterns_matched=patterns_matched,
                        modules_blocked=modules_blocked
                    )
                    
                    return {
                        'success': False,
                        'error': f"Security violation: {len(violations)} issues found",
                        'violations': [{'description': v.description, 'severity': v.severity.name} for v in violations]
                    }
                
                # Execute code
                result = session.run(code, libraries=libraries)
                
                # Log successful execution
                self.monitor.log_security_event(
                    event_type="execution",
                    severity=SecurityIssueSeverity.SAFE,
                    user_id=user_id,
                    session_id=session_id,
                    details={
                        'code_length': len(code),
                        'libraries': libraries or [],
                        'exit_code': getattr(result, 'exit_code', 0)
                    },
                    code_hash=code_hash
                )
                
                return {
                    'success': True,
                    'result': {
                        'stdout': getattr(result, 'stdout', ''),
                        'stderr': getattr(result, 'stderr', ''),
                        'exit_code': getattr(result, 'exit_code', 0)
                    }
                }
        
        except Exception as e:
            # Log execution error
            self.monitor.log_security_event(
                event_type="error",
                severity=SecurityIssueSeverity.MEDIUM,
                user_id=user_id,
                session_id=session_id,
                details={'error': str(e)},
                code_hash=code_hash
            )
            
            return {
                'success': False,
                'error': f"Execution failed: {str(e)}"
            }


def demo_security_monitoring():
    """Demonstrate security monitoring system."""
    print("🔍 Security Monitoring Dashboard Demo")
    print("=====================================")
    
    # Create security policy
    policy = SecurityPolicy(
        safety_level=SecurityIssueSeverity.MEDIUM,
        patterns=[
            SecurityPattern(
                pattern=r"\bos\.system\s*\(",
                description="System command execution",
                severity=SecurityIssueSeverity.HIGH,
            ),
            SecurityPattern(
                pattern=r"\beval\s*\(",
                description="Dynamic evaluation",
                severity=SecurityIssueSeverity.MEDIUM,
            ),
        ],
        dangerous_modules=[
            DangerousModule("os", "OS interface", SecurityIssueSeverity.HIGH),
            DangerousModule("subprocess", "Subprocess", SecurityIssueSeverity.HIGH),
        ]
    )
    
    # Initialize monitoring
    monitor = SecurityMonitor("demo_security.db")
    monitor.start_monitoring()
    
    # Create secure execution wrapper
    executor = SecureExecutionWrapper(policy, monitor)
    
    # Simulate various code executions
    test_scenarios = [
        ("print('Hello, World!')", "user1", "session1"),  # Safe
        ("import os\nos.system('ls')", "user2", "session2"),  # Violation
        ("import math\nprint(math.sqrt(16))", "user1", "session3"),  # Safe
        ("eval('2 + 2')", "user3", "session4"),  # Violation
        ("import requests\nrequests.get('http://example.com')", "user2", "session5"),  # Safe (depending on policy)
    ]
    
    print("\n📊 Executing test scenarios...")
    for i, (code, user_id, session_id) in enumerate(test_scenarios, 1):
        print(f"\nScenario {i}: User {user_id}")
        print(f"Code: {code[:50]}...")
        
        result = executor.execute_with_monitoring(code, user_id, session_id)
        
        if result['success']:
            print("✅ Execution successful")
        else:
            print(f"❌ Execution blocked: {result['error']}")
    
    # Wait a moment for monitoring to process
    time.sleep(2)
    
    # Display dashboard data
    dashboard_data = monitor.get_dashboard_data()
    
    print("\n📈 Security Dashboard Summary:")
    metrics = dashboard_data['metrics']
    print(f"   Total Executions: {metrics['total_executions']}")
    print(f"   Blocked Executions: {metrics['blocked_executions']}")
    print(f"   Violation Rate: {metrics['violation_rate']:.2%}")
    
    if metrics['top_violations']:
        print(f"   Top Violations:")
        for violation in metrics['top_violations'][:3]:
            print(f"     - {violation['pattern']}: {violation['count']} times")
    
    print(f"   Severity Distribution: {metrics['severity_distribution']}")
    
    # Show recent events
    recent_events = dashboard_data['recent_events']
    print(f"\n📋 Recent Events ({len(recent_events)} total):")
    for event in recent_events[-5:]:  # Last 5 events
        print(f"   {event['timestamp'][:19]} - {event['event_type'].upper()} - {event['severity']}")
        if event['user_id']:
            print(f"     User: {event['user_id']}, Session: {event['session_id']}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    print("\n✅ Security monitoring demonstration completed!")


if __name__ == "__main__":
    try:
        demo_security_monitoring()
    except KeyboardInterrupt:
        print("\n👋 Monitoring demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise
