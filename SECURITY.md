# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within RMR, please send an email to the project maintainers at **[INSERT SECURITY EMAIL]**. All security vulnerabilities will be promptly addressed.

**Please do not report security vulnerabilities through public GitHub issues.**

### What to Include

When reporting a vulnerability, please include:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** assessment
4. **Suggested fix** (if available)
5. **Your contact information** for follow-up

### Response Expectations

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours.
- **Assessment**: We will investigate and assess the vulnerability within 5 business days.
- **Resolution**: We will work on a fix and coordinate disclosure with you.
- **Disclosure**: We will publicly disclose the vulnerability after a fix is available.

### Disclosure Policy

We follow coordinated disclosure:

1. **Reporter** reports vulnerability privately
2. **Maintainers** acknowledge and investigate
3. **Fix** is developed and tested
4. **Release** with fix is published
5. **Public disclosure** after users have time to update

### Security Best Practices

When using RMR in production:

1. **Dependencies**: Keep dependencies updated
2. **Environment**: Use secure environment variables for sensitive configuration
3. **Data**: Ensure training data is from trusted sources
4. **Models**: Validate model inputs and outputs
5. **Deployment**: Follow secure deployment practices

## Security-Related Configuration

RMR does not handle sensitive user data by default. However, if you are:

- Using custom datasets with sensitive information
- Deploying models in production environments
- Integrating with external services

Please follow your organization's security policies and best practices.

## Contact

For security-related inquiries, please contact:

- **Email**: [INSERT SECURITY EMAIL]
- **GitHub Security**: Use GitHub's private vulnerability reporting feature

Thank you for helping keep RMR and its users safe.
