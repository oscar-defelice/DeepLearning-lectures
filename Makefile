run:
	docker build -t oscardefelice/deep-learning-lectures .
	docker run --name deep-learning-lectures \
	-v ${PWD}:/home/jovyan/work \
	-p 8888:8888 \
	oscardefelice/deep-learning-lectures
