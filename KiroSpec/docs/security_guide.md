# Security Configuration Guide

This guide covers the comprehensive security features of the LangChain Vector Database, including authentication, authorization, encryption, and monitoring.

## Table of Contents

1. [Security Overview](#security-overview)
2. [Authentication Configuration](#authentication-configuration)
3. [Authorization and RBAC](#authorization-and-rbac)
4. [Data Encryption](#data-encryption)
5. [PII Detection and Masking](#pii-detection-and-masking)
6. [Rate Limiting](#rate-limiting)
7. [Security Monitoring](#security-monitoring)
8. [Audit Logging](#audit-logging)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Security Overview

The LangChain Vector Database provides enterprise-grade security features:

- **Authentication**: API key and JWT token support
- **Authorization**: Role-based access control (RBAC)
- **Encryption**: Data at rest and in transit
- **PII Protection**: Automatic detection and masking
- **Rate Limiting**: Protection against abuse
- **Monitoring**: Real-time security event tracking
- **Audit Logging**: Comprehensive operation logging

## Authentication Configuration

### Basic Security Setup

```python
from langchain_vector_db.models.config import VectorDBConfig, SecurityConfig

security_config = SecurityConfig(
    auth_enabled=True,
    auth_type="api_key",  # or "jwt"
    rbac_enabled=True,
    encryption_enabled=True,
    audit_logging_enabled=True
)

config = VectorDBConfig(
    storage_type="local",
    security=security_config
)
```

### API Key Authentication

```python
# Create API keys for users
manager = VectorDatabaseManager(config)
security_manager = manager.security_manager

# Create API key with specific roles
api_key = security_manager.create_api_key(
    user_id="john_doe",
    roles=["writer", "reader"],
    expires_hours=24
)

# Authenticate user
auth_token = security_manager.authenticate_api_key(
    api_key=api_key,
    user_id="john_doe",
    ip_address="192.168.1.100"
)
```

### JWT Token Authentication

```python
security_config = SecurityConfig(
    auth_enabled=True,
    auth_type="jwt",
    jwt_secret_key="your-secret-key",
    jwt_algorithm="HS256",
    jwt_expiration_hours=8
)

# Create JWT token
jwt_token = security_manager.create_jwt_token(
    user_id="jane_doe",
    roles=["admin"],
    custom_claims={"department": "engineering"}
)

# Authenticate with JWT
auth_token = security_manager.authenticate_jwt_token(jwt_token)
```

## Authorization and RBAC

### Role Definitions

The system supports the following built-in roles:

- **admin**: Full access to all operations
- **writer**: Can create, read, update, and delete documents
- **reader**: Can only read documents and perform searches
- **viewer**: Can only perform searches (no document access)

### Custom Roles

```python
# Define custom role with specific permissions
custom_permissions = [
    "documents.read",
    "documents.create",
    "search.query",
    "search.advanced"
]

api_key = security_manager.create_api_key(
    user_id="analyst",
    roles=["custom_analyst"],
    custom_permissions=custom_permissions
)
```

### Permission Checking

```python
# Check if user has specific permission
has_permission = security_manager.authorize_operation(
    auth_token=auth_token,
    resource="documents",
    action="create"
)

if has_permission:
    # Perform operation
    doc_ids = manager.add_documents(documents, auth_token=auth_token)
```

## Data Encryption

### Encryption at Rest

```python
security_config = SecurityConfig(
    encryption_enabled=True,
    encryption_algorithm="AES-256-GCM",
    encryption_key_rotation_days=90
)

# Data is automatically encrypted when stored
doc_ids = manager.add_documents(documents, auth_token=auth_token)
```

### Encryption in Transit

```python
# Enable TLS for all communications
security_config = SecurityConfig(
    tls_enabled=True,
    tls_cert_path="/path/to/cert.pem",
    tls_key_path="/path/to/key.pem",
    tls_ca_path="/path/to/ca.pem"
)
```

### Manual Encryption/Decryption

```python
# Encrypt sensitive data manually
sensitive_data = b"Confidential information"
encrypted_data = security_manager.encrypt_data(sensitive_data)

# Decrypt when needed
decrypted_data = security_manager.decrypt_data(encrypted_data)
```

## PII Detection and Masking

### Automatic PII Detection

```python
security_config = SecurityConfig(
    pii_detection_enabled=True,
    pii_detection_confidence=0.8,
    auto_mask_pii=True
)

# PII is automatically detected and masked
text_with_pii = "Contact John at john.doe@company.com or 555-123-4567"
pii_matches = security_manager.detect_pii(text_with_pii)

for match in pii_matches:
    print(f"Found {match.type}: {match.value} at position {match.start}-{match.end}")
```

### Custom PII Patterns

```python
# Add custom PII detection patterns
custom_patterns = {
    "employee_id": r"EMP-\d{6}",
    "project_code": r"PROJ-[A-Z]{3}-\d{4}"
}

security_manager.add_pii_patterns(custom_patterns)
```

### Data Masking

```python
# Mask sensitive data
original_text = "Employee EMP-123456 works on PROJ-ABC-1234"
masked_text = security_manager.mask_sensitive_data(original_text)
# Result: "Employee [EMPLOYEE_ID] works on [PROJECT_CODE]"
```

## Rate Limiting

### Basic Rate Limiting

```python
security_config = SecurityConfig(
    rate_limiting_enabled=True,
    max_requests_per_minute=100,
    max_requests_per_hour=1000,
    rate_limit_by_ip=True,
    rate_limit_by_user=True
)
```

### Custom Rate Limits

```python
# Set custom rate limits for specific users
security_manager.set_user_rate_limit(
    user_id="power_user",
    requests_per_minute=500,
    requests_per_hour=5000
)

# Set rate limits for specific operations
security_manager.set_operation_rate_limit(
    operation="similarity_search",
    requests_per_minute=200
)
```

### Rate Limit Monitoring

```python
# Check current rate limit status
rate_limit_status = security_manager.get_rate_limit_status(
    user_id="john_doe",
    ip_address="192.168.1.100"
)

print(f"Requests remaining: {rate_limit_status['requests_remaining']}")
print(f"Reset time: {rate_limit_status['reset_time']}")
```

## Security Monitoring

### Real-time Monitoring

```python
security_config = SecurityConfig(
    security_monitoring_enabled=True,
    intrusion_detection_enabled=True,
    brute_force_threshold=5,
    suspicious_activity_threshold=10
)
```

### Security Alerts

```python
# Get security alerts
alerts = security_manager.get_security_alerts(severity="high")

for alert in alerts:
    print(f"Alert: {alert['alert_type']}")
    print(f"Description: {alert['description']}")
    print(f"Timestamp: {alert['timestamp']}")
    print(f"Severity: {alert['severity']}")
```

### Threat Detection

```python
# Get threat indicators
threats = security_manager.get_threat_indicators()

for threat in threats:
    print(f"Threat: {threat['type']}")
    print(f"Source: {threat['source_ip']}")
    print(f"Confidence: {threat['confidence']}")
```

### IP Blocking

```python
# Block suspicious IP addresses
security_manager.block_ip(
    ip_address="192.168.1.200",
    reason="Multiple failed authentication attempts",
    duration_hours=24
)

# Check if IP is blocked
is_blocked = security_manager.is_ip_blocked("192.168.1.200")
```

## Audit Logging

### Comprehensive Audit Trail

```python
security_config = SecurityConfig(
    audit_logging_enabled=True,
    audit_log_level="INFO",
    audit_log_format="json",
    audit_log_retention_days=365
)
```

### Audit Event Types

The system logs the following events:

- Authentication attempts (success/failure)
- Authorization checks
- Document operations (CRUD)
- Search operations
- Configuration changes
- Security events
- System access

### Retrieving Audit Events

```python
# Get recent audit events
audit_events = security_manager.get_audit_events(
    limit=100,
    start_time=datetime.utcnow() - timedelta(hours=24),
    event_types=["authentication", "document_access"]
)

for event in audit_events:
    print(f"Time: {event['timestamp']}")
    print(f"User: {event['user_id']}")
    print(f"Operation: {event['operation']}")
    print(f"Result: {event['result']}")
    print(f"IP: {event['ip_address']}")
```

### Audit Event Filtering

```python
# Filter audit events by criteria
filtered_events = security_manager.get_audit_events(
    user_id="john_doe",
    operation="add_documents",
    result="success",
    ip_address="192.168.1.100"
)
```

## Best Practices

### 1. Authentication

- Use strong API keys (minimum 32 characters)
- Implement key rotation policies
- Set appropriate expiration times
- Use JWT for stateless authentication

### 2. Authorization

- Follow principle of least privilege
- Use role-based access control
- Regularly review user permissions
- Implement resource-level authorization

### 3. Encryption

- Enable encryption for all sensitive data
- Use strong encryption algorithms (AES-256)
- Implement proper key management
- Rotate encryption keys regularly

### 4. PII Protection

- Enable automatic PII detection
- Implement data masking policies
- Regular PII pattern updates
- Train staff on PII handling

### 5. Monitoring

- Enable comprehensive logging
- Set up real-time alerts
- Monitor for suspicious activities
- Regular security assessments

### 6. Rate Limiting

- Set appropriate rate limits
- Monitor for abuse patterns
- Implement progressive penalties
- Use both IP and user-based limits

## Troubleshooting

### Common Issues

#### Authentication Failures

```python
# Debug authentication issues
try:
    auth_token = security_manager.authenticate_api_key(api_key, user_id)
except AuthenticationException as e:
    print(f"Authentication failed: {e}")
    
    # Check if API key is valid
    is_valid = security_manager.validate_api_key(api_key)
    print(f"API key valid: {is_valid}")
    
    # Check if API key is expired
    key_info = security_manager.get_api_key_info(api_key)
    print(f"Key expires: {key_info.get('expires_at')}")
```

#### Permission Denied

```python
# Debug authorization issues
has_permission = security_manager.authorize_operation(
    auth_token, "documents", "create"
)

if not has_permission:
    print(f"User roles: {auth_token.roles}")
    print(f"User permissions: {auth_token.permissions}")
    
    # Check required permissions
    required_perms = security_manager.get_required_permissions(
        "documents", "create"
    )
    print(f"Required permissions: {required_perms}")
```

#### Rate Limiting Issues

```python
# Debug rate limiting
try:
    manager.add_documents(documents, auth_token=auth_token)
except VectorDBException as e:
    if "rate limit" in str(e).lower():
        status = security_manager.get_rate_limit_status(
            user_id=auth_token.user_id
        )
        print(f"Rate limit status: {status}")
        print(f"Reset in: {status['reset_time'] - datetime.utcnow()}")
```

### Security Configuration Validation

```python
# Validate security configuration
validation_result = security_manager.validate_security_config()

if not validation_result.is_valid:
    print("Security configuration issues:")
    for issue in validation_result.issues:
        print(f"- {issue.severity}: {issue.message}")
```

### Security Health Check

```python
# Perform security health check
security_health = security_manager.get_security_health_status()

print(f"Overall security health: {security_health['status']}")
print(f"Security score: {security_health['score']}/100")

for check, result in security_health['checks'].items():
    status = "✅" if result['passed'] else "❌"
    print(f"{status} {check}: {result['message']}")
```

## Security Compliance

### GDPR Compliance

```python
# Implement right to be forgotten
def delete_user_data(user_id):
    # Find all documents by user
    user_docs = manager.find_documents_by_metadata({"user_id": user_id})
    
    # Delete documents
    for doc in user_docs:
        manager.delete_document(doc.doc_id, permanent=True)
    
    # Remove from audit logs (if legally required)
    security_manager.anonymize_audit_logs(user_id)
```

### SOC 2 Compliance

```python
# Generate compliance reports
compliance_report = security_manager.generate_compliance_report(
    standard="SOC2",
    start_date=datetime.utcnow() - timedelta(days=90),
    end_date=datetime.utcnow()
)

print(f"Compliance score: {compliance_report['score']}")
print(f"Controls passed: {compliance_report['controls_passed']}")
print(f"Controls failed: {compliance_report['controls_failed']}")
```

This security guide provides comprehensive coverage of all security features. Refer to the API documentation for detailed method signatures and additional configuration options.