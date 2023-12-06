run:
	mpic++ main.cpp -o main.o &&  mpirun -np 30 --oversubscribe ./main.o


graph:
	python3 main.py

.PHONE: run graph