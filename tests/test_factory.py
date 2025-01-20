"""Tests for session factory functionality."""

import pytest
from unittest.mock import MagicMock, patch

import docker
from kubernetes import client as k8s_client

from llm_sandbox.factory import (
    UnifiedSessionFactory,
    DockerSessionFactory,
    KubernetesSessionFactory,
    PodmanSessionFactory
)
from llm_sandbox.exceptions import ValidationError
from llm_sandbox.monitoring import ResourceLimits

@pytest.fixture
def mock_docker_client():
    return MagicMock(spec=docker.DockerClient)

@pytest.fixture
def mock_k8s_client():
    return MagicMock(spec=k8s_client.CoreV1Api)

@pytest.fixture
def resource_limits():
    return ResourceLimits(
        max_cpu_percent=50.0,
        max_memory_bytes=256 * 1024 * 1024
    )

@pytest.fixture
def docker_factory(mock_docker_client, resource_limits):
    return DockerSessionFactory(
        client=mock_docker_client,
        default_resource_limits=resource_limits
    )

@pytest.fixture
def k8s_factory(mock_k8s_client):
    return KubernetesSessionFactory(
        client=mock_k8s_client,
        default_namespace="test-namespace"
    )

@pytest.fixture
def unified_factory(mock_docker_client, mock_k8s_client, resource_limits):
    return UnifiedSessionFactory(
        docker_client=mock_docker_client,
        k8s_client=mock_k8s_client,
        default_resource_limits=resource_limits,
        default_k8s_namespace="test-namespace"
    )

def test_docker_session_factory(docker_factory):
    session = docker_factory.create_session(
        image="python:3.9",
        lang="python",
        verbose=True
    )
    
    assert session is not None
    assert session.image == "python:3.9"
    assert session.lang == "python"

def test_kubernetes_session_factory(k8s_factory):
    session = k8s_factory.create_session(
        image="python:3.9",
        lang="python",
        verbose=True
    )
    
    assert session is not None
    assert session.kube_namespace == "test-namespace"

def test_podman_session_factory():
    factory = PodmanSessionFactory()
    
    session = factory.create_session(
        image="python:3.9",
        lang="python",
        verbose=True
    )
    
    assert session is not None

@pytest.mark.parametrize("backend,expected_type", [
    ("docker", "SandboxDockerSession"),
    ("kubernetes", "SandboxKubernetesSession"),
    ("podman", "SandboxPodmanSession")
])
def test_unified_factory_backends(unified_factory, backend, expected_type):
    session = unified_factory.create_session(
        backend=backend,
        image="python:3.9",
        lang="python"
    )
    
    assert session is not None
    assert expected_type in str(type(session))

def test_unified_factory_invalid_backend(unified_factory):
    with pytest.raises(ValidationError) as exc_info:
        unified_factory.create_session(
            backend='invalid',
            image="python:3.9",
            lang="python"
        )
    
    assert "Unsupported backend" in str(exc_info.value)

def test_docker_session_factory_resource_limits(docker_factory, resource_limits):
    session = docker_factory.create_session(
        image="python:3.9",
        lang="python"
    )
    
    assert 'cpu_count' in session.container_configs
    assert 'mem_limit' in session.container_configs
    assert session.container_configs['cpu_count'] == resource_limits.max_cpu_percent / 100.0

def test_kubernetes_session_factory_custom_namespace(k8s_factory):
    session = k8s_factory.create_session(
        image="python:3.9",
        lang="python",
        kube_namespace="custom-namespace"
    )
    
    assert session.kube_namespace == "custom-namespace"

@pytest.mark.parametrize("namespace", [
    "default",
    "custom-ns",
    "test-environment"
])
def test_kubernetes_session_factory_namespaces(mock_k8s_client, namespace):
    factory = KubernetesSessionFactory(
        client=mock_k8s_client,
        default_namespace=namespace
    )
    
    session = factory.create_session(
        image="python:3.9",
        lang="python"
    )
    
    assert session.kube_namespace == namespace

@patch('kubernetes.config')
def test_kubernetes_session_factory_no_client(mock_k8s_config):
    factory = KubernetesSessionFactory()
    
    session = factory.create_session(
        image="python:3.9",
        lang="python"
    )
    
    mock_k8s_config.load_kube_config.assert_called_once()
    assert session is not None 