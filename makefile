all:
	nvcc generator/generator.cu visualizer/visualizer.cu -o MandelbrotSet -lsfml-graphics -lsfml-window -lsfml-system
