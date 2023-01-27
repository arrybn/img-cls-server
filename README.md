Sample implementation of Image classification HTTP server based on OpenVINO inference runtime.

The server consists of http server (Poco), inference server (based on OpenVINO), data processing queue and result database.

The server can handle parallel connections, inference is done in several parallel threads.

## Build
* <code>git checkout git@github.com:arrybn/img-cls-server.git</code>
* <code>cd img-cls-server</code>
* <code>docker build . -t img-cls-server:latest</code>
* <code>docker push <your-docker-registry>/img-cls-server:latest</code>

## Deploy
* copy <code>docker-compose.yml</code> to the host, on which you'd like to run
* ssh to the host, go to the directory with copied <code>docker-compose.yml</code>
* edit <code>docker-compose.yml</code> to make <code>image:</code> value matching your docker registry and host port matching some available port (<code>9090</code> dy default)
* <code>docker-compose up -d</code>

## Inference

Python:
```python
import requests
file = {'img': open('n03063599_coffee_mug.jpeg', 'rb')}
requests.post("http://130.61.122.219:9090", files=file).text
```
Terminal:
```shell
curl -F img=@pizaa.jpeg http://130.61.122.219:9090
```


## Development
I recommend using Visual Studio code and DevContainers extension to develop and build the code inside the container while having pretty GUI