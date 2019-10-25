import sys
import os
import docker
import tempfile
import tarfile

cur_dir = os.path.dirname(os.path.abspath(__file__))

if sys.version_info < (3, 0):
    try:
        from cStringIO import StringIO
    except ImportError:
        from StringIO import StringIO
    PY3 = False
else:
    from io import BytesIO as StringIO
    PY3 = True

def build_model(docker_client, 
				name,
				version,
                model_path,
				prediction_file,
                port=7000,
                base_image="alice97/serve-base",
                container_registry=None,
                pkgs_to_install=None):
	version = str(version)
	run_cmd = ''
	if pkgs_to_install:
		run_as_lst = 'RUN pip install'.split(' ')
		run_cmd = ' '.join(run_as_lst + pkgs_to_install)
	with tempfile.NamedTemporaryFile(mode="w+b", suffix="tar") as context_file:
		# Create build context tarfile
		with tarfile.TarFile(fileobj=context_file, mode="w") as context_tar:
			context_tar.add(model_path)
			try:
				df_contents = StringIO(
					str.encode(
						"FROM {container_name}\n{run_command}\n COPY {model_path} /model\n WORKDIR /model\n EXPOSE {port}\n CMD [ \"python3\", \"./{prediction_file}\" ]".
						format(
							container_name=base_image,
							model_path=model_path,
							prediction_file=prediction_file,
							run_command=run_cmd,
							port=port)))
				df_tarinfo = tarfile.TarInfo('Dockerfile')
				df_contents.seek(0, os.SEEK_END)
				df_tarinfo.size = df_contents.tell()
				df_contents.seek(0)
				context_tar.addfile(df_tarinfo, df_contents)
			except TypeError:
				df_contents = StringIO(
					"FROM {container_name}\n{run_command}\nCOPY {model_path} /model WORKDIR /model EXPOSE {port} CMD ['\"python\"','\"./{prediction_file}\"']".
					format(
						container_name=base_image,
						model_path=model_path,
						prediction_file=prediction_file,
						run_command=run_cmd,
						port=port))
				df_tarinfo = tarfile.TarInfo('Dockerfile')
				df_contents.seek(0, os.SEEK_END)
				df_tarinfo.size = df_contents.tell()
				df_contents.seek(0)
				context_tar.addfile(df_tarinfo, df_contents)
		# Exit Tarfile context manager to finish the tar file
		# Seek back to beginning of file for reading
		context_file.seek(0)
		image = "{name}:{version}".format(
			name=name, version=version)
		print("Building model Docker image with model file from {}".format(prediction_file))
		image_result, build_logs = docker_client.images.build(fileobj=context_file, custom_context=True, tag=image)

	return image

def run_container(docker_client, image, cmd=None, name=None, ports=None,
				labels=None, environment=None, log_config=None, volumes=None,
				user=None):
	return docker_client.containers.run(
		image,
		command=cmd,
		name=name,
		ports=ports,
		labels=labels,
		environment=environment,
		volumes=volumes,
		user=user,
		log_config=log_config)


if __name__ == "__main__":
	name = "pytorch_container"
	version = 1
	model_path = os.path.join(cur_dir,"pytorch_container/")
	prediction_file = "pytorch_container.py"
	ports = {'7000': 7000}

	docker_client = docker.from_env()
	image = build_model(docker_client, 
						name,
						version,
						model_path,
		                prediction_file,
		                port=7000,
		                base_image="alice97/serve-base",
		                container_registry=None,
		                pkgs_to_install=None)
	print("Create image successfully!")
	run_container(docker_client, image, ports=ports)
