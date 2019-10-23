# rpc
A workplace for developing rpc communication service between server and containers

# build docker image for contanier
```
cd container/
docker build --tag noop-container-app .
```

# run docker image
```
docker run --name noop-container --env-file env.list -p 7000:7000 noop-container-app
```

# start rpc client(another terminal)
```
python client_test.py
```
