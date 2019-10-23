import sys
import docker

if __name__ == "__main__":
	docker_client = docker.from_env()
	path = "./"
	tag = "noop_container"
	ports = {'7000': 7000}

	image_result, build_logs = docker_client.images.build(path=path, tag=tag)
	print("Create image successfully!")
	docker_client.containers.run(tag, ports=ports)
