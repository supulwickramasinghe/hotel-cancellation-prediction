# Guest Shield — Visual Documentation

The architecture diagrams and screenshots are linked from the repository root `README.md`.

## Architecture
- `architecture/ml-pipeline.png` and `.svg`: offline training, clustering, classification, evaluation, and model export.
- `architecture/aws-serverless-architecture.png` and `.svg`: IAM-protected inference request, ECR container delivery, Lambda, SAM/CloudFormation, and CloudWatch.

## Deployment screenshots
- `screenshots/lambda-overview.png`: function overview and configuration.
- `screenshots/api-gateway-iam.png`: IAM-secured POST route.
- `screenshots/ecr-container.png`: deployed image repository.
- `screenshots/cloudformation-stack.png`: provisioned stack resources.
- `screenshots/cloudwatch-monitoring.png`: test-period operational metrics.
- `screenshots/prediction-response.png`: successful signed API call.

These images are evidence from a development deployment. Redacted identifiers should remain redacted when replacing screenshots.
