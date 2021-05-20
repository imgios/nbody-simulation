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
        free(commBuf);
    } else {
        // Init the work buffer with the size received from master
        workBuf = (Body*)malloc(sizeof(Body) * sendcount[rank]);
    }

    /* const int nIters = (argv[2] != NULL) ? atoi(argv[2]) : 10; // Simulation iterations
    const float dt = 0.01f; // Time step
    // Buffer used for gathered data
    float *tempBuf = (float*)malloc(bytes);
    Body *gatherBuf = (Body*)tempBuf;
    for (int iter = 1; iter <= nIters; i++) {
        // Synchronize all cores before starting iteration time
        MPI_Barrier(MPI_COMM_WORLD);
        double iterStart = MPI_Wtime();

        // Send particles array to all cores with a Broadcast
        MPI_Bcast(buf, nBodies, bodytype, 0, MPI_COMM_WORLD);

        // Every core will compute its own portion
        bodyForce(buf, displacements[rank], sendcount[rank], dt, nBodies);

        // Gather bodies computed by every single core
        MPI_Gatherv(buf + displacements[rank], sendcount[rank], bodytype, gatherBuf, sendcount, displacements, bodytype, 0, MPI_COMM_WORLD);

        // Synchronize all cores before taking iteration time
        MPI_Barrier(MPI_COMM_WORLD);
        double iterEnd = MPI_Wtime();

        if (rank == 0) {
            buf = gatherBuf; // Master saves the gathered data inside the buffer before the next iteration
            printf("Iteration #%d completed in %f seconds.\n", iter, iterEnd - iterStart);
        }
    } */

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all cores
    end = MPI_Wtime();
    if (rank == 0) { // Master
        printf("Simulation completed in %f seconds.\n", end - start);
    }
    MPI_Finalize();
}