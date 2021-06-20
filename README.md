# Building Image

docker build -t coach-neuro-ml-app:"version" .

# Run Image

docker run "image id" -p 5000:8000

# Tag Image

docker tag "image id" gcr.io/coachneuro-dev/coach-neuro-ml-app

# Push Image

docker push gcr.io/coachneuro-dev/coach-neuro-ml-app
