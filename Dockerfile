
# AWS Lambda Python base image.
FROM public.ecr.aws/lambda/python:3.13

# Install inference dependencies.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt

RUN pip install --no-cache-dir \
    -r ${LAMBDA_TASK_ROOT}/requirements.txt \
    --target ${LAMBDA_TASK_ROOT}

# Copy application code.
COPY app/ ${LAMBDA_TASK_ROOT}/app/

# Copy trained model.
COPY models/guest_shield_model.joblib \
    ${LAMBDA_TASK_ROOT}/models/guest_shield_model.joblib

# Tell Lambda which function to execute.
CMD ["app.handler.lambda_handler"]
