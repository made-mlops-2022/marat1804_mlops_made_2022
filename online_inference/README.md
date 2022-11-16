docker build -t online_inference:v1 .

docker run --name online_inference -p 8000:8000 online_inference:v1