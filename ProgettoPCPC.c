
#include <stdio.h>
#include <string.h>
#include <math.h>
#include<stdlib.h>
#include "mpi.h"

#define MAX 100
#define NSTEPS 100

int main(int argc, char* argv[]){
	long int n=500, m=500;
	if(argc==3){
		n=strtol(argv[1], NULL, 10);
		m=strtol(argv[2], NULL, 10);

		if(n<=0)
			n=500;
		if(m<=0)
			m=500;
	}
	int  my_rank; /* rank of process */
	int  p, q, r;       /* number of processes */
	MPI_Status status ;   /* return status for receive */
	float *x, *y;
	float **xrow;
	short stop;
	double begin, end;
	MPI_Request req;

	/* start up MPI */

	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);


	x=malloc(n*m*sizeof(float));
	xrow=malloc(n*sizeof(float*));
	for(int i=0;i<n;i++)
		xrow[i]=x+(i*m);

	//inizialization
	srand(1);
	for(int i=0; i<n*m; i++){
		x[i]=rand() % MAX;
	}

	if(my_rank==0){
		begin=MPI_Wtime();
	}

	int indexes[2];
	if(my_rank==0){
		q=(n-2)/p;
		r=(n-2)%p;
		for(int i=1;i<p;i++){
			indexes[0] = (i<r?(i*q)+i:(i*q)+r)+1;
			indexes[1]=i<r?indexes[0]+q:indexes[0]+q-1;
			MPI_Send(indexes, 2, MPI_INT, i, 0, MPI_COMM_WORLD);
		}
		indexes[0]=1;
		indexes[1]=r>0?q+1:q;
	}
	else{
		MPI_Recv(indexes, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
	}

	y=malloc((indexes[1]-indexes[0]+1)*(m-1)*sizeof(float));

	int step=0;
	stop=0;
	while(step<NSTEPS && !stop){
		if(p>1){
			if(my_rank==0){
				MPI_Isend(xrow[indexes[1]],m,MPI_INT,1,0,MPI_COMM_WORLD, &req);
				MPI_Recv(xrow[indexes[1]+1],m,MPI_INT,1,0,MPI_COMM_WORLD, &status);
			}
			else if(my_rank==p-1){
				MPI_Isend(xrow[indexes[0]], m, MPI_INT, my_rank-1,0,MPI_COMM_WORLD,&req);
				MPI_Recv(xrow[indexes[0]-1],m,MPI_INT,my_rank-1,0,MPI_COMM_WORLD, &status);
			}
			else{
				MPI_Isend(xrow[indexes[0]], m, MPI_INT, my_rank-1,0,MPI_COMM_WORLD,&req);
				MPI_Isend(xrow[indexes[1]],m,MPI_INT,my_rank+1,0,MPI_COMM_WORLD,&req);
				MPI_Recv(xrow[indexes[0]-1],m,MPI_INT,my_rank-1,0,MPI_COMM_WORLD, &status);
				MPI_Recv(xrow[indexes[1]+1],m,MPI_INT,my_rank+1,0,MPI_COMM_WORLD, &status);
			}
		}

		float *k=y;
		for(int i=indexes[0];i<=indexes[1];i++){
			for(int j=1;j<m-1;j++){
				*k=(xrow[i+1][j] + xrow[i-1][j] + xrow[i][j+1] + xrow[i][j-1])/4;
				k++;
			}
		}
		float diffnorm = 0;
		k=y;
		for(int i=indexes[0];i<=indexes[1];i++)
			for(int j=1;j<m-1;j++){
				diffnorm += (*k - xrow[i][j]) * (*k - xrow[i][j]);
				k++;
			}
		if(my_rank==0){
			float res;
			for(int i=1;i<p;i++){
				MPI_Recv(&res,1,MPI_FLOAT,i,0,MPI_COMM_WORLD, &status);
				diffnorm+=res;
			}
			diffnorm=sqrt(diffnorm);
			//printf("%f\n",diffnorm);
			if(diffnorm<0.01f)
				stop=1;
			else
				stop=0;
			for(int i=1;i<p;i++)
				MPI_Isend(&stop,1,MPI_SHORT,i,0,MPI_COMM_WORLD, &req);
		}

		else{
			MPI_Isend(&diffnorm, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req);
			MPI_Recv(&stop, 1 ,MPI_SHORT, 0, 0, MPI_COMM_WORLD, &status);
		}

		k=y;
		for(int i=indexes[0];i<=indexes[1];i++)
			for(int j=1;j<m-1;j++){
				xrow[i][j]=*k;
				k++;
			}
		step++;
	}

	if(my_rank==0){
		int start=indexes[1]+1;
		int length=0;
		for(int i=1;i<p; i++){
			length=i<r?q+1:q;
			MPI_Recv(xrow[start], length*m,MPI_FLOAT,i,0,MPI_COMM_WORLD, &status);
			start+=length;
		}
		end=MPI_Wtime();
		printf("computed in %f ms\n", (end-begin)*1000);
		if(stop)
			printf("converge in %d passi\n\n", step);
		/*FILE *f=fopen("result.txt", "w");
		for(int i=1;i<N-1;i++){
			for(int j=1;j<M-1;j++){
				fprintf(f,"%f\t", xrow[i][j]);
			}
			fprintf(f,"\n");
		}
		fprintf(f,"\n");*/
	}
	else{
		MPI_Send(xrow[indexes[0]], (indexes[1]-indexes[0]+1)*m, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();


	return 0;
}
