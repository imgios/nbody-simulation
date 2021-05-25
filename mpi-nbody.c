#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define SOFTENING 1e-9f
#define MASTER 0

// Struct for Bodies
typedef struct {
    float x, y, z; // position
    float vx, vy, vz; // velocity
} Body;

/*
* This function calculates the body force of every element of the array.
*
*   p: array of bodies
*   dt: delta
*   n: number of total bodies
*/
void bodyForce(Body *p, float dt, int n) {
  for (int i = 0; i < n; i++) { 
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx;
    p[i].vy += dt*Fy;
    p[i].vz += dt*Fz;
  }
}

/*
* This function calculates the body force of a certain array,
* relating to other bodies.
*
*   particles: array of bodies to compute
*   bodies: number of bodies to compute
*   relatedParticles: array of related bodies
*   relatedBodies: number of related bodies
*/
void relatedBodyForce(Body *particles, int bodies, Body *relatedParticles, int relatedBodies, float dt) {
    for (int i = 0; i < bodies; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < relatedBodies) {
            float dx = relatedParticles[j].x - particles[i].x;
            float dy = relatedParticles[j].y - particles[i].y;
            float dz = relatedParticles[j].z - particles[i].z;
            float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        particles[i].vx += dt*Fx;
        particles[i].vy += dt*Fy;
        particles[i].vz += dt*Fz;
    }
}

/*
* This function randomizes body position and velocity data.
* It's mainly used for the array initialization.
*
*   data: array of bodies
*   n: number of bodies
*/
void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

int main (int argc, int ** argv) {
    int numtasks, rank, tag = 1;
    // Get the number of bodies from input,
    // and if not specified set it to 30000
    int nBodies = (argv[1] != NULL) ? atoi(argv[1]) : 30000;
    Body *commBuf = NULL;
    Body *workBuf = NULL;
    // Vars used for time elapsed during computation
    double start, end;

    MPI_Status status;
    MPI_Request requests[numtasks];

    // Init MPI env
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (numtasks == 1) {
        // Implement or not the case when the code is ran by a single core?
        printf("Execute the code with 2 or more cores.\n");
        return 0;
    }

    // Derivated Struct Datatype
    MPI_Datatype bodytype, oldtypes[1];
    MPI_Aint offsets[1];
    int blockcounts[1];
    offsets[0] = 0;
    oldtypes[0] = MPI_FLOAT;
    blockcounts[0] = 6;
    MPI_Type_create_struct(1, blockcounts, offsets, oldtypes, &bodytype);
    MPI_Type_commit(&bodytype);

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all cores
    start = MPI_Wtime();
    int sendcount[numtasks]; // Contains how many items every rank should receive
    int displacements[numtasks]; // Contains the offset for every rank

    if (rank == MASTER) { // master
        int bytes = nBodies * sizeof(Body);
        float *particlesBuf = (float*)malloc(bytes);
        Body *commBuf = (Body*)particlesBuf;
        // Init bodies position and velocity data
        randomizeBodies(commBuf, 6*nBodies);

        for (int i = 0; i < numtasks; i++) {
            int count = nBodies/numtasks;
            if (i != numtasks - 1) { // Not the last core
                sendcount[i] = count;
            } else { // Last core
                // If count module > 0 then we give more particles to the last core
                // If count module = 0 then we give the same amount of particles to all cores
                sendcount[i] = nBodies - ((numtasks - 1) * (count));
            }
        }

        // Send item counts to all cores
        MPI_Bcast(sendcount, numtasks, MPI_INT, MASTER, MPI_COMM_WORLD);

        displacements[0] = 0; // Master starts from index 0
        for (int i = 1; i < numtasks; i++) {
            // Init the displacement using the number of items sent and the index used
            // by the previous core.
            displacements[i] = sendcount[i-1] + displacements[i-1];
        }

        // Split the data array to all cores using a Scatterv
        MPI_Scatterv(&commBuf[0], sendcount, displacements, bodytype, &workBuf[0], sendcount[rank], bodytype, MASTER, MPI_COMM_WORLD);

        // Store only its own particles
        workBuf = (Body*)malloc(sizeof(Body) * sendcount[MASTER]);
        for (int = 0; i < sendcount[MASTER]; i++) {
            workBuf[i] = commBuf[i];
        }

        // Release unused memory
        free(particlesBuf);
    } else {
        // Init the work buffer with the size received from master
        workBuf = (Body*)malloc(sizeof(Body) * sendcount[rank]);
    }

    // Allocate memory for related bodies, useful for the second stage of the computation
    int relatedBodies = nBodies - sendcount[rank]; // total bodies - own bodies
    Body *relatedParticles = (Body*)malloc(sizeof(Body) * relatedBodies);

    // Simulation iterations
    const int nIters = (argv[2] != NULL) ? atoi(argv[2]) : 10;

    // Start computing
    for (int iter = 0; iter < nIters; iter++) {
        // Sync all cores before starting
        MPI_Barrier(MPI_COMM_WORLD);
        int iterStart = MPI_Wtime();
        const float dt = 0.01f; // Time step
        // Send own particles to all cores
        MPI_Ibcast(workBuf, sendcount[rank], bodytype, rank, MPI_COMM_WORLD, &requests[rank]);

        // Retrieve related particles from other cores
        int relatedIndex = 0;
        for (int i = 0; i < numtasks; i++) {
            if (i != rank) {
                // Retrieve the offset to start storing related particles
                // in order to avoid overwrite
                int relatedOffset =  relatedIndex * sendcount[i-1];
                if (i == 0) {
                    relatedOffset = 0;
                }
                // Receive related particles from rank i
                MPI_Ibcast(&relatedParticles[relatedOffset], sendcount[i], bodytype, i, MPI_COMM_WORLD, &requests[i]);
                // Increment relatedIndex in order to calculate different offsets
                relatedIndex++;
            }
        }

        // Catch all the request sent from other cores and compute
        for (int requestsCount = 0; requestsCount < numtasks - 1; requestsCount++) {
            int reqIndex;
            MPI_Waitany(numtasks, requests, &reqIndex, &status);
            // Check if the request is not sent by the same core who is receiving data
            if (reqIndex != rank) {
                int bodiescount = sendcount[reqIndex];
                if (reqIndex > rank) {
                    reqIndex += -1;
                }
                int startOffset = reqIndex * bodiescount;
                // Compute body force for own particles
                relatedBodyForce(workBuf, sendcount[rank], &relatedParticles[startOffset], bodiescount);
            }
        }
        
        // Integrate position for own particles
        for (int i = 0 ; i < sendcount[rank]; i++) { // integrate position
            workBuf[i].x += workBuf[i].vx*dt;
            workBuf[i].y += workBuf[i].vy*dt;
            workBuf[i].z += workBuf[i].vz*dt;
        }

        // Sync all cores to take iteration time
        MPI_Barrier(MPI_COMM_WORLD);
        int iterEnd = MPI_Wtime();
        if (rank == MASTER) {
            // Master will print a string that indicates the iteration completition
            printf("Iteration #%d completed in %d seconds.", iter + 1, iterEnd - iterStart);
            // Clear the output buffer and move the buffered data to the console
            fflush(stdout);
        }
    }

    if (rank == MASTER) { // Master must gather all particles and show results
        // Retrieve the number of bytes to allocate
        int bytes = nBodies * sizeof(Body);
        // Allocate the communication buffer in order to contain all particles
        commBuf = (Body*)malloc(bytes);
    }

    // Gather all particles to the Master's communication buffer
    MPI_Gatherv(workBuf, sendcount[rank], bodytype, commBuf, sendcount, displacements, bodytype, MASTER, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all cores
    end = MPI_Wtime();
    if (rank == 0) { // Master
        printf("Simulation completed in %f seconds.\n", end - start);
        double avgTime = (end - start)/(double)(nIters);
        printf("Avg. iteration time: %f seconds\n", avgTime);
        fflush(stdout);
    }

    // Cleanup
    MPI_Type_free(&bodytype);
    free(workBuf);
    if (rank == MASTER) {
        free(commBuf);
    }

    MPI_Finalize();
}