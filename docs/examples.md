# Examples

This section provides practical examples of using LLM Sandbox in various scenarios.

## Basic Examples

### Hello World

```python
from llm_sandbox import SandboxSession

# Simple code execution
with SandboxSession(lang="python") as session:
    result = session.run("print('Hello, LLM Sandbox!')")
    print(result.stdout)  # Output: Hello, LLM Sandbox!
```

### Multi-Language Support

```python
# Execute code in different languages
languages_code = {
    "python": 'print("Hello from Python!")',
    "javascript": 'console.log("Hello from JavaScript!");',
    "java": '''
        public class Hello {
            public static void main(String[] args) {
                System.out.println("Hello from Java!");
            }
        }
    ''',
    "cpp": '''
        #include <iostream>
        int main() {
            std::cout << "Hello from C++!" << std::endl;
            return 0;
        }
    ''',
    "go": '''
        package main
        import "fmt"
        func main() {
            fmt.Println("Hello from Go!")
        }
    ''',
    "ruby": 'puts "Hello from Ruby!"'
}

for lang, code in languages_code.items():
    with SandboxSession(lang=lang) as session:
        result = session.run(code)
        print(f"{lang}: {result.stdout.strip()}")
```

## Data Science Examples

### Data Analysis Pipeline

```python
from llm_sandbox import SandboxSession

data_analysis_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate sample data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.normal(35, 10, n_samples),
    'income': np.random.lognormal(10.5, 0.5, n_samples),
    'education_years': np.random.randint(12, 21, n_samples),
    'job_satisfaction': np.random.uniform(1, 10, n_samples)
})

# Add derived features
data['income_category'] = pd.cut(
    data['income'], 
    bins=[0, 30000, 60000, 100000, np.inf],
    labels=['Low', 'Medium', 'High', 'Very High']
)

# Basic statistics
print("Dataset Overview:")
print(f"Shape: {data.shape}")
print(f"\nColumns: {list(data.columns)}")
print("\nStatistical Summary:")
print(data.describe())

# Correlation analysis
corr_matrix = data[['age', 'income', 'education_years', 'job_satisfaction']].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Age distribution
axes[0, 0].hist(data['age'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# 2. Income by category
data['income_category'].value_counts().plot(kind='bar', ax=axes[0, 1])
axes[0, 1].set_title('Income Categories')
axes[0, 1].set_xlabel('Category')
axes[0, 1].set_ylabel('Count')

# 3. Education vs Income scatter
axes[1, 0].scatter(data['education_years'], data['income'], alpha=0.5)
axes[1, 0].set_title('Education vs Income')
axes[1, 0].set_xlabel('Education Years')
axes[1, 0].set_ylabel('Income')

# 4. Job satisfaction boxplot by income category
data.boxplot(column='job_satisfaction', by='income_category', ax=axes[1, 1])
axes[1, 1].set_title('Job Satisfaction by Income Category')

plt.tight_layout()
plt.show()

# Additional insights
print("\nKey Insights:")
avg_satisfaction_by_income = data.groupby('income_category')['job_satisfaction'].mean()
print(f"Average job satisfaction by income category:")
for category, satisfaction in avg_satisfaction_by_income.items():
    print(f"  {category}: {satisfaction:.2f}")
"""

with SandboxSession(lang="python") as session:
    result = session.run(
        data_analysis_code,
        libraries=["pandas", "numpy", "matplotlib", "seaborn"]
    )
    print(result.stdout)
```

### Machine Learning Workflow

```python
from llm_sandbox import ArtifactSandboxSession
import base64

ml_code = """
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

results = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    
    # Test predictions
    y_pred = model.predict(X_test_scaled)
    
    # Store results
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': (y_pred == y_test).mean(),
        'predictions': y_pred
    }
    
    print(f"\n{name} Results:")
    print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print(f"Test Accuracy: {results[name]['test_accuracy']:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Model comparison
model_names = list(results.keys())
test_accuracies = [results[m]['test_accuracy'] for m in model_names]
axes[0, 0].bar(model_names, test_accuracies)
axes[0, 0].set_title('Model Test Accuracies')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_ylim(0, 1)

# 2. Cross-validation scores
cv_means = [results[m]['cv_mean'] for m in model_names]
cv_stds = [results[m]['cv_std'] for m in model_names]
axes[0, 1].bar(model_names, cv_means, yerr=cv_stds, capsize=5)
axes[0, 1].set_title('Cross-Validation Scores')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_ylim(0, 1)

# 3. Confusion matrix for best model
best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
conf_matrix = confusion_matrix(y_test, best_model[1]['predictions'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_title(f'Confusion Matrix - {best_model[0]}')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

# 4. Feature importance (for Random Forest)
rf_model = models['Random Forest']
feature_importance = rf_model.feature_importances_
top_features_idx = np.argsort(feature_importance)[-10:][::-1]
axes[1, 1].barh(range(10), feature_importance[top_features_idx])
axes[1, 1].set_yticks(range(10))
axes[1, 1].set_yticklabels([f'Feature {i}' for i in top_features_idx])
axes[1, 1].set_title('Top 10 Feature Importances (Random Forest)')
axes[1, 1].set_xlabel('Importance')

plt.tight_layout()
plt.show()

print(f"\nBest Model: {best_model[0]} with test accuracy: {best_model[1]['test_accuracy']:.3f}")
"""

# Execute and capture plots
with ArtifactSandboxSession(lang="python") as session:
    result = session.run(
        ml_code,
        libraries=["scikit-learn", "pandas", "numpy", "matplotlib", "seaborn"]
    )
    
    print("Model training output:")
    print(result.stdout)
    
    # Save plots
    for i, plot in enumerate(result.plots):
        with open(f"ml_analysis_plot_{i}.png", "wb") as f:
            f.write(base64.b64decode(plot.content_base64))
        print(f"Saved plot: ml_analysis_plot_{i}.png")
```

## Web Scraping Examples

### Basic Web Scraping

```python
from llm_sandbox import SandboxSession
from llm_sandbox.security import get_security_policy

web_scraping_code = """
import requests
from bs4 import BeautifulSoup
import json
import time

# Example: Scrape Python.org news
url = "https://www.python.org/news/"

try:
    # Make request with headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Educational Bot)'
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract news items
    news_items = []
    for article in soup.find_all('div', class_='list-recent-posts')[:5]:
        title_elem = article.find('h3')
        date_elem = article.find('time')
        
        if title_elem and date_elem:
            news_items.append({
                'title': title_elem.text.strip(),
                'date': date_elem.get('datetime', ''),
                'link': title_elem.find('a')['href'] if title_elem.find('a') else ''
            })
    
    # Display results
    print(f"Latest Python News ({len(news_items)} items):")
    print("=" * 50)
    for item in news_items:
        print(f"Title: {item['title']}")
        print(f"Date: {item['date']}")
        print(f"Link: https://www.python.org{item['link']}")
        print("-" * 50)
    
    # Save as JSON
    with open('/tmp/python_news.json', 'w') as f:
        json.dump(news_items, f, indent=2)
    print("\nNews saved to /tmp/python_news.json")
    
except requests.RequestException as e:
    print(f"Error fetching data: {e}")
except Exception as e:
    print(f"Error processing data: {e}")
"""

# Use web scraping security policy
with SandboxSession(
    lang="python",
    security_policy=get_security_policy("web_scraping")
) as session:
    result = session.run(
        web_scraping_code,
        libraries=["requests", "beautifulsoup4"]
    )
    print(result.stdout)
```

### API Data Collection

```python
api_integration_code = """
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Example: Fetch and analyze GitHub repository data
# Using public API (no auth required for basic queries)

def fetch_repo_data(owner, repo):
    """Fetch repository information from GitHub API"""
    base_url = "https://api.github.com"
    
    # Repository info
    repo_url = f"{base_url}/repos/{owner}/{repo}"
    repo_response = requests.get(repo_url)
    repo_data = repo_response.json()
    
    # Recent commits
    commits_url = f"{base_url}/repos/{owner}/{repo}/commits"
    commits_response = requests.get(commits_url, params={'per_page': 30})
    commits_data = commits_response.json()
    
    return repo_data, commits_data

# Fetch data for a popular repository
owner, repo = "python", "cpython"
repo_info, recent_commits = fetch_repo_data(owner, repo)

print(f"Repository: {repo_info['full_name']}")
print(f"Description: {repo_info['description']}")
print(f"Stars: {repo_info['stargazers_count']:,}")
print(f"Forks: {repo_info['forks_count']:,}")
print(f"Open Issues: {repo_info['open_issues_count']:,}")

# Analyze commit patterns
commit_dates = []
for commit in recent_commits:
    if isinstance(commit, dict) and 'commit' in commit:
        date_str = commit['commit']['author']['date']
        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        commit_dates.append(date)

# Create commit frequency visualization
if commit_dates:
    # Group by day
    df = pd.DataFrame({'date': commit_dates})
    df['day'] = df['date'].dt.date
    daily_commits = df.groupby('day').size()
    
    plt.figure(figsize=(12, 6))
    daily_commits.plot(kind='bar')
    plt.title(f'Daily Commit Frequency - {repo_info["full_name"]}')
    plt.xlabel('Date')
    plt.ylabel('Number of Commits')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Commit statistics
    print(f"\nCommit Statistics (last {len(recent_commits)} commits):")
    print(f"Average commits per day: {len(commit_dates) / 30:.2f}")
    print(f"Most recent commit: {max(commit_dates).strftime('%Y-%m-%d %H:%M:%S')}")
"""

with SandboxSession(lang="python") as session:
    result = session.run(
        api_integration_code,
        libraries=["requests", "pandas", "matplotlib"]
    )
    print(result.stdout)
```

## LLM Integration Examples

### Code Generation and Testing

```python
from llm_sandbox import SandboxSession
import openai

class CodeGeneratorWithTesting:
    """Generate and test code using LLM + Sandbox"""
    
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_code(self, task: str) -> str:
        """Generate code for a given task"""
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Python code generator. Generate clean, efficient code."},
                {"role": "user", "content": f"Write Python code to: {task}"}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    
    def test_code(self, code: str, test_cases: list) -> dict:
        """Test generated code with test cases"""
        test_results = []
        
        with SandboxSession(lang="python") as session:
            # First, define the function/code
            setup_result = session.run(code)
            if setup_result.exit_code != 0:
                return {
                    "success": False,
                    "error": "Code compilation/setup failed",
                    "details": setup_result.stderr
                }
            
            # Run test cases
            for i, test in enumerate(test_cases):
                test_code = f"""
# Test case {i + 1}
try:
    result = {test['input']}
    expected = {test['expected']}
    passed = result == expected
    print(f"Test {i + 1}: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  Expected: {expected}")
        print(f"  Got: {result}")
except Exception as e:
    print(f"Test {i + 1}: ERROR - {e}")
"""
                result = session.run(test_code)
                test_results.append({
                    "test_id": i + 1,
                    "output": result.stdout,
                    "passed": "PASS" in result.stdout
                })
        
        return {
            "success": True,
            "test_results": test_results,
            "all_passed": all(t["passed"] for t in test_results)
        }

# Example usage
generator = CodeGeneratorWithTesting(api_key="your-api-key")

# Generate function
task = "create a function that calculates the nth Fibonacci number efficiently"
generated_code = generator.generate_code(task)

print("Generated Code:")
print(generated_code)
print("\n" + "="*50 + "\n")

# Test the generated code
test_cases = [
    {"input": "fibonacci(0)", "expected": 0},
    {"input": "fibonacci(1)", "expected": 1},
    {"input": "fibonacci(5)", "expected": 5},
    {"input": "fibonacci(10)", "expected": 55},
]

test_results = generator.test_code(generated_code, test_cases)
print("Test Results:")
for result in test_results["test_results"]:
    print(result["output"])
```

### Interactive Coding Assistant

```python
from llm_sandbox import SandboxSession
import json

class InteractiveCodingAssistant:
    """Interactive assistant that can write and execute code"""
    
    def __init__(self):
        self.session = None
        self.context = {}
        self.history = []
    
    def start_session(self, language="python"):
        """Start a persistent coding session"""
        self.session = SandboxSession(lang=language)
        self.session.open()
        print(f"Started {language} session")
    
    def execute(self, code: str, description: str = ""):
        """Execute code and maintain context"""
        if not self.session:
            self.start_session()
        
        result = self.session.run(code)
        
        # Store in history
        self.history.append({
            "description": description,
            "code": code,
            "output": result.stdout,
            "error": result.stderr,
            "success": result.exit_code == 0
        })
        
        return result
    
    def save_context(self, name: str, value: any):
        """Save a value to context"""
        self.context[name] = value
    
    def get_history_summary(self):
        """Get a summary of execution history"""
        return [{
            "step": i + 1,
            "description": h["description"],
            "success": h["success"],
            "output_preview": h["output"][:100] + "..." if len(h["output"]) > 100 else h["output"]
        } for i, h in enumerate(self.history)]
    
    def close_session(self):
        """Close the coding session"""
        if self.session:
            self.session.close()
            print("Session closed")

# Example: Interactive data analysis session
assistant = InteractiveCodingAssistant()

# Step 1: Load data
result1 = assistant.execute("""
import pandas as pd
import numpy as np

# Create sample dataset
np.random.seed(42)
data = pd.DataFrame({
    'product': ['A', 'B', 'C', 'D'] * 25,
    'sales': np.random.randint(100, 1000, 100),
    'date': pd.date_range('2024-01-01', periods=100, freq='D')
})
print(f"Dataset created with {len(data)} rows")
print(data.head())
""", "Create sample sales dataset")

# Step 2: Analyze data
result2 = assistant.execute("""
# Aggregate by product
product_summary = data.groupby('product')['sales'].agg(['sum', 'mean', 'std'])
print("Product Summary:")
print(product_summary)
print(f"\nBest performing product: {product_summary['sum'].idxmax()}")
""", "Analyze sales by product")

# Step 3: Create visualization
result3 = assistant.execute("""
import matplotlib.pyplot as plt

# Sales over time by product
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Line plot
for product in data['product'].unique():
    product_data = data[data['product'] == product]
    ax1.plot(product_data['date'], product_data['sales'], label=product, alpha=0.7)
ax1.set_title('Sales Over Time by Product')
ax1.set_xlabel('Date')
ax1.set_ylabel('Sales')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# Bar plot of total sales
product_summary['sum'].plot(kind='bar', ax=ax2)
ax2.set_title('Total Sales by Product')
ax2.set_xlabel('Product')
ax2.set_ylabel('Total Sales')

plt.tight_layout()
plt.show()
""", "Create sales visualizations")

# Get session summary
print("\nSession Summary:")
for step in assistant.get_history_summary():
    print(f"Step {step['step']}: {step['description']} - {'✓' if step['success'] else '✗'}")
    if step['output_preview']:
        print(f"  Output: {step['output_preview']}")

assistant.close_session()
```

## Security Examples

### Secure Code Execution Service

```python
from llm_sandbox import SandboxSession
from llm_sandbox.security import (
    SecurityPolicy, 
    SecurityPattern, 
    DangerousModule,
    SecurityIssueSeverity,
    get_security_policy
)
import hashlib
import time
from typing import Dict, Any

class SecureCodeExecutionService:
    """Production-ready secure code execution service"""
    
    def __init__(self):
        self.execution_log = []
        self.rate_limits = {}  # user_id -> [timestamps]
        self.blocked_patterns = set()
        
        # Create custom security policy
        self.security_policy = self._create_security_policy()
    
    def _create_security_policy(self) -> SecurityPolicy:
        """Create comprehensive security policy"""
        # Start with production preset
        policy = get_security_policy("production")
        
        # Add custom patterns
        custom_patterns = [
            SecurityPattern(
                pattern=r"\b(curl|wget)\s+http",
                description="External download attempt",
                severity=SecurityIssueSeverity.HIGH
            ),
            SecurityPattern(
                pattern=r"\b(127\.0\.0\.1|localhost)\b",
                description="Localhost access attempt",
                severity=SecurityIssueSeverity.MEDIUM
            ),
            SecurityPattern(
                pattern=r"\bbase64\s*\.\s*b64decode",
                description="Base64 decoding (potential obfuscation)",
                severity=SecurityIssueSeverity.MEDIUM
            )
        ]
        
        for pattern in custom_patterns:
            policy.add_pattern(pattern)
        
        return policy
    
    def check_rate_limit(self, user_id: str, max_per_minute: int = 10) -> bool:
        """Check if user is within rate limits"""
        current_time = time.time()
        
        # Clean old entries
        if user_id in self.rate_limits:
            self.rate_limits[user_id] = [
                t for t in self.rate_limits[user_id] 
                if current_time - t < 60
            ]
        else:
            self.rate_limits[user_id] = []
        
        # Check limit
        if len(self.rate_limits[user_id]) >= max_per_minute:
            return False
        
        self.rate_limits[user_id].append(current_time)
        return True
    
    def execute_code(
        self, 
        code: str, 
        user_id: str,
        language: str = "python",
        timeout: int = 10,
        max_memory: str = "128m"
    ) -> Dict[str, Any]:
        """Execute code with comprehensive security checks"""
        
        # Rate limiting
        if not self.check_rate_limit(user_id):
            return {
                "success": False,
                "error": "Rate limit exceeded. Please try again later."
            }
        
        # Input validation
        if not code or len(code) > 50000:
            return {
                "success": False,
                "error": "Invalid code length"
            }
        
        # Log execution attempt
        execution_id = hashlib.sha256(
            f"{user_id}{time.time()}{code}".encode()
        ).hexdigest()[:16]
        
        self.execution_log.append({
            "id": execution_id,
            "user_id": user_id,
            "timestamp": time.time(),
            "language": language,
            "code_hash": hashlib.sha256(code.encode()).hexdigest()
        })
        
        try:
            with SandboxSession(
                lang=language,
                security_policy=self.security_policy,
                runtime_configs={
                    "timeout": timeout,
                    "mem_limit": max_memory,
                    "cpu_count": 1,
                    "network_mode": "none",  # No network access
                    "read_only": True,  # Read-only root filesystem
                    "user": "nobody:nogroup"  # Run as nobody
                }
            ) as session:
                # Security check
                is_safe, violations = session.is_safe(code)
                
                if not is_safe:
                    # Log security violation
                    self.execution_log[-1]["security_violation"] = True
                    
                    return {
                        "success": False,
                        "error": "Code failed security checks",
                        "violations": [
                            {"description": v.description, "severity": v.severity.name}
                            for v in violations
                        ]
                    }
                
                # Execute code
                result = session.run(code)
                
                # Prepare response
                response = {
                    "success": result.exit_code == 0,
                    "execution_id": execution_id,
                    "output": result.stdout[:5000],  # Limit output size
                    "error": result.stderr[:1000] if result.stderr else None,
                    "exit_code": result.exit_code,
                    "truncated": len(result.stdout) > 5000
                }
                
                # Update log
                self.execution_log[-1]["success"] = response["success"]
                
                return response
                
        except TimeoutError:
            return {
                "success": False,
                "error": "Execution timeout exceeded"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}"
            }
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get execution statistics for a user"""
        user_executions = [
            log for log in self.execution_log 
            if log["user_id"] == user_id
        ]
        
        if not user_executions:
            return {"total_executions": 0}
        
        success_count = sum(1 for e in user_executions if e.get("success", False))
        violation_count = sum(1 for e in user_executions if e.get("security_violation", False))
        
        return {
            "total_executions": len(user_executions),
            "successful_executions": success_count,
            "security_violations": violation_count,
            "success_rate": success_count / len(user_executions),
            "last_execution": max(e["timestamp"] for e in user_executions)
        }

# Example usage
service = SecureCodeExecutionService()

# Safe code execution
result1 = service.execute_code(
    code="print('Hello, World!')",
    user_id="user123"
)
print("Safe code result:", result1)

# Unsafe code attempt
result2 = service.execute_code(
    code="import os\nos.system('ls')",
    user_id="user123"
)
print("Unsafe code result:", result2)

# Get user stats
stats = service.get_user_statistics("user123")
print("User statistics:", stats)
```

## File Processing Examples

### CSV Processing Pipeline

```python
from llm_sandbox import SandboxSession
import base64

csv_processing_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Sample CSV data
csv_data = """Date,Product,Region,Sales,Quantity
2024-01-01,Widget A,North,1500,50
2024-01-01,Widget B,North,2300,75
2024-01-01,Widget A,South,1200,40
2024-01-02,Widget A,North,1600,55
2024-01-02,Widget B,South,2100,70
2024-01-02,Widget C,North,1800,60
2024-01-03,Widget A,South,1400,45
2024-01-03,Widget B,North,2400,80
2024-01-03,Widget C,South,1900,65
"""

# Load data
df = pd.read_csv(StringIO(csv_data))
df['Date'] = pd.to_datetime(df['Date'])

print("Data Overview:")
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

# Data transformations
df['Revenue_per_unit'] = df['Sales'] / df['Quantity']
df['Day_of_week'] = df['Date'].dt.day_name()

# Aggregations
print("\n" + "="*50)
print("ANALYSIS RESULTS")
print("="*50)

# 1. Sales by Product
product_sales = df.groupby('Product')['Sales'].agg(['sum', 'mean', 'count'])
print("\nSales by Product:")
print(product_sales)

# 2. Regional Performance
regional_stats = df.groupby('Region').agg({
    'Sales': ['sum', 'mean'],
    'Quantity': 'sum',
    'Revenue_per_unit': 'mean'
}).round(2)
print("\nRegional Performance:")
print(regional_stats)

# 3. Time-based Analysis
daily_sales = df.groupby('Date')['Sales'].sum()
print(f"\nDaily Sales Trend:")
for date, sales in daily_sales.items():
    print(f"  {date.strftime('%Y-%m-%d')}: ${sales:,.2f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Sales by Product (Bar)
product_sales['sum'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Total Sales by Product')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].set_xlabel('Product')

# 2. Regional Distribution (Pie)
region_sales = df.groupby('Region')['Sales'].sum()
axes[0, 1].pie(region_sales.values, labels=region_sales.index, autopct='%1.1f%%')
axes[0, 1].set_title('Sales Distribution by Region')

# 3. Daily Sales Trend (Line)
daily_sales.plot(ax=axes[1, 0], marker='o', color='green')
axes[1, 0].set_title('Daily Sales Trend')
axes[1, 0].set_ylabel('Sales ($)')
axes[1, 0].set_xlabel('Date')

# 4. Revenue per Unit by Product (Box)
df.boxplot(column='Revenue_per_unit', by='Product', ax=axes[1, 1])
axes[1, 1].set_title('Revenue per Unit Distribution')
axes[1, 1].set_ylabel('Revenue per Unit ($)')

plt.tight_layout()
plt.show()

# Export processed data
processed_df = df.copy()
processed_df['Date'] = processed_df['Date'].dt.strftime('%Y-%m-%d')
processed_csv = processed_df.to_csv(index=False)

print("\n" + "="*50)
print("PROCESSED DATA (First 5 rows):")
print(processed_df.head())

# Save to file
with open('/tmp/processed_sales_data.csv', 'w') as f:
    f.write(processed_csv)
print("\nProcessed data saved to /tmp/processed_sales_data.csv")
"""

with SandboxSession(lang="python") as session:
    result = session.run(
        csv_processing_code,
        libraries=["pandas", "matplotlib"]
    )
    print(result.stdout)
    
    # Retrieve processed file
    session.copy_from_runtime(
        "/tmp/processed_sales_data.csv",
        "./processed_sales_data.csv"
    )
    print("\nProcessed CSV file downloaded successfully!")
```

### Image Processing

```python
image_processing_code = """
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import io

# Create a sample image
width, height = 400, 300
image = Image.new('RGB', (width, height), 'white')
pixels = image.load()

# Generate gradient pattern
for x in range(width):
    for y in range(height):
        r = int((x / width) * 255)
        g = int((y / height) * 255)
        b = 128
        pixels[x, y] = (r, g, b)

print("Original image created")

# Apply various transformations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Original
axes[0, 0].imshow(image)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

# 2. Blur
blurred = image.filter(ImageFilter.BLUR)
axes[0, 1].imshow(blurred)
axes[0, 1].set_title('Blur Filter')
axes[0, 1].axis('off')

# 3. Edge Detection
edges = image.filter(ImageFilter.FIND_EDGES)
axes[0, 2].imshow(edges)
axes[0, 2].set_title('Edge Detection')
axes[0, 2].axis('off')

# 4. Brightness Adjustment
enhancer = ImageEnhance.Brightness(image)
brightened = enhancer.enhance(1.5)
axes[1, 0].imshow(brightened)
axes[1, 0].set_title('Increased Brightness')
axes[1, 0].axis('off')

# 5. Contrast Adjustment
contrast_enhancer = ImageEnhance.Contrast(image)
high_contrast = contrast_enhancer.enhance(2.0)
axes[1, 1].imshow(high_contrast)
axes[1, 1].set_title('High Contrast')
axes[1, 1].axis('off')

# 6. Grayscale
grayscale = image.convert('L')
axes[1, 2].imshow(grayscale, cmap='gray')
axes[1, 2].set_title('Grayscale')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# Image statistics
img_array = np.array(image)
print(f"\nImage Statistics:")
print(f"Dimensions: {img_array.shape}")
print(f"Mean RGB values: R={img_array[:,:,0].mean():.1f}, G={img_array[:,:,1].mean():.1f}, B={img_array[:,:,2].mean():.1f}")
print(f"Min/Max values: {img_array.min()}/{img_array.max()}")

# Save processed images
image.save('/tmp/original.png')
blurred.save('/tmp/blurred.png')
edges.save('/tmp/edges.png')
grayscale.save('/tmp/grayscale.png')

print("\nImages saved to /tmp/")
"""

with SandboxSession(lang="python") as session:
    result = session.run(
        image_processing_code,
        libraries=["pillow", "matplotlib", "numpy"]
    )
    print(result.stdout)
```

## Performance Optimization Examples

### Parallel Processing

```python
parallel_processing_code = """
import multiprocessing as mp
import time
import numpy as np
from functools import partial

def compute_intensive_task(x, power=2):
    """Simulate CPU-intensive computation"""
    result = 0
    for i in range(1000):
        result += np.sin(x) ** power + np.cos(x) ** power
    return result

def parallel_map(func, data, num_processes=None):
    """Parallel map implementation"""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(func, data)
    
    return results

# Generate test data
data_sizes = [1000, 5000, 10000]
results = {}

for size in data_sizes:
    data = np.random.rand(size)
    
    # Sequential processing
    start_time = time.time()
    sequential_results = [compute_intensive_task(x) for x in data]
    sequential_time = time.time() - start_time
    
    # Parallel processing
    start_time = time.time()
    parallel_results = parallel_map(compute_intensive_task, data)
    parallel_time = time.time() - start_time
    
    # Store results
    results[size] = {
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'speedup': sequential_time / parallel_time,
        'efficiency': (sequential_time / parallel_time) / mp.cpu_count()
    }
    
    print(f"\nData size: {size}")
    print(f"Sequential time: {sequential_time:.3f}s")
    print(f"Parallel time: {parallel_time:.3f}s")
    print(f"Speedup: {results[size]['speedup']:.2f}x")
    print(f"Efficiency: {results[size]['efficiency']:.2%}")

# Visualization
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Execution times
sizes = list(results.keys())
seq_times = [results[s]['sequential_time'] for s in sizes]
par_times = [results[s]['parallel_time'] for s in sizes]

x = np.arange(len(sizes))
width = 0.35

ax1.bar(x - width/2, seq_times, width, label='Sequential')
ax1.bar(x + width/2, par_times, width, label='Parallel')
ax1.set_xlabel('Data Size')
ax1.set_ylabel('Time (seconds)')
ax1.set_title('Execution Time Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(sizes)
ax1.legend()

# Speedup
speedups = [results[s]['speedup'] for s in sizes]
ax2.plot(sizes, speedups, 'o-', color='green', linewidth=2, markersize=8)
ax2.axhline(y=mp.cpu_count(), color='r', linestyle='--', label=f'Ideal ({mp.cpu_count()}x)')
ax2.set_xlabel('Data Size')
ax2.set_ylabel('Speedup')
ax2.set_title('Parallel Processing Speedup')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nSystem info: {mp.cpu_count()} CPU cores available")
"""

with SandboxSession(
    lang="python",
    runtime_configs={"cpu_count": 4}  # Limit CPU cores
) as session:
    result = session.run(
        parallel_processing_code,
        libraries=["numpy", "matplotlib"]
    )
    print(result.stdout)
```

## Error Handling Examples

### Robust Error Handling

```python
error_handling_code = """
import sys
import traceback
import logging
from typing import Optional, Union, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SafeExecutor:
    """Safe code executor with comprehensive error handling"""
    
    @staticmethod
    def safe_divide(a: float, b: float) -> Optional[float]:
        """Safe division with error handling"""
        try:
            if b == 0:
                logger.warning(f"Division by zero attempted: {a} / {b}")
                return None
            return a / b
        except TypeError as e:
            logger.error(f"Type error in division: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in division: {e}")
            return None
    
    @staticmethod
    def safe_file_read(filename: str) -> Optional[str]:
        """Safe file reading with multiple fallbacks"""
        try:
            with open(filename, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"File not found: {filename}")
            # Try alternative locations
            alt_locations = [f"/tmp/{filename}", f"./{filename}"]
            for alt in alt_locations:
                try:
                    with open(alt, 'r') as f:
                        logger.info(f"Found file at alternative location: {alt}")
                        return f.read()
                except:
                    continue
            return None
        except PermissionError:
            logger.error(f"Permission denied reading file: {filename}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading file: {e}")
            return None
    
    @staticmethod
    def safe_json_parse(json_str: str) -> Optional[dict]:
        """Safe JSON parsing with error details"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error at position {e.pos}: {e.msg}")
            # Try to fix common issues
            if "'" in json_str:
                logger.info("Attempting to fix single quotes...")
                try:
                    fixed_json = json_str.replace("'", '"')
                    return json.loads(fixed_json)
                except:
                    pass
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing JSON: {e}")
            return None
    
    @staticmethod
    def safe_execute(func, *args, **kwargs) -> tuple[bool, Any, Optional[str]]:
        """Execute function safely with full error capture"""
        try:
            result = func(*args, **kwargs)
            return True, result, None
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            error_trace = traceback.format_exc()
            
            logger.error(f"{error_type}: {error_msg}")
            logger.debug(f"Full traceback:\n{error_trace}")
            
            return False, None, f"{error_type}: {error_msg}"

# Test error handling
executor = SafeExecutor()

print("Testing Error Handling:")
print("=" * 50)

# Test 1: Division
print("\n1. Division Tests:")
test_cases = [(10, 2), (10, 0), ("10", 2), (10, "zero")]
for a, b in test_cases:
    result = executor.safe_divide(a, b)
    print(f"  {a} / {b} = {result}")

# Test 2: File operations
print("\n2. File Operation Tests:")
files = ["exists.txt", "not_exists.txt", "/root/restricted.txt"]

# Create test file
with open("/tmp/exists.txt", "w") as f:
    f.write("Test content")

for filename in files:
    content = executor.safe_file_read(filename)
    print(f"  Read '{filename}': {'Success' if content else 'Failed'}")

# Test 3: JSON parsing
print("\n3. JSON Parsing Tests:")
json_tests = [
    '{"valid": true}',
    "{'invalid': True}",  # Single quotes
    '{"incomplete": ',     # Incomplete
    'not json at all'      # Invalid
]

for json_str in json_tests:
    result = executor.safe_json_parse(json_str)
    print(f"  Parse '{json_str[:20]}...': {'Success' if result else 'Failed'}")

# Test 4: Generic function execution
print("\n4. Generic Function Execution:")

def risky_function(x):
    if x < 0:
        raise ValueError("Negative value not allowed")
    if x == 0:
        raise ZeroDivisionError("Cannot process zero")
    return 1 / x

test_values = [10, 0, -5, "string"]
for val in test_values:
    success, result, error = executor.safe_execute(risky_function, val)
    print(f"  risky_function({val}): {'Success' if success else f'Failed - {error}'}")

print("\n" + "=" * 50)
print("Error handling tests completed!")
"""

with SandboxSession(lang="python") as session:
    result = session.run(error_handling_code)
    print(result.stdout)
```

## Next Steps

- Learn about [Security Best Practices](security.md)
- Explore [API Reference](api-reference.md)
- Check [Configuration Options](configuration.md)
- Read about [Contributing](contributing.md)