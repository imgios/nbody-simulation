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

        for (int j = 0; j < relatedBodies; j++) {
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

int main (int argc, char ** argv) {
    int numtasks, rank, tag = 1;
    // Get the number of bodies from input,
    // and if not specified set it to 30000
    int nBodies = (argv[1] != NULL) ? atoi(argv[1]) : 30000;
    float *particlesBuf;
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
        particlesBuf = (float*)malloc(bytes);
        commBuf = (Body*)particlesBuf;
        // Init bodies position and velocity data
        randomizeBodies(particlesBuf, 6*nBodies);

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
    }

    // Send item counts to all cores
    MPI_Bcast(sendcount, numtasks, MPI_INT, MASTER, MPI_COMM_WORLD);

    // Init the work buffer with the size received from master
    workBuf = (Body*)malloc(sizeof(Body) * sendcount[rank]);

    if (rank == MASTER) {
        // Compute offsets
        displacements[0] = 0; // Master starts from index 0
        for (int i = 1; i < numtasks; i++) {
            // Init the displacement using the number of items sent and the index used
            // by the previous core.
            displacements[i] = sendcount[i-1] + displacements[i-1];
        }
    }

    // Master split the data array to all cores using a Scatterv
    MPI_Scatterv(&commBuf[0], sendcount, displacements, bodytype, &workBuf[0], sendcount[rank], bodytype, MASTER, MPI_COMM_WORLD);

    if (rank == MASTER) {
        // Store only its own particles
        for (int i = 0; i < sendcount[MASTER]; i++) {
            workBuf[i] = commBuf[i];
        }

        // Release unused memory
        free(particlesBuf);
        commBuf = NULL;
    }

    // Allocate memory for related bodies, useful for the second stage of the computation
    int relatedBodies = nBodies - sendcount[rank]; // total bodies - own bodies
    Body *relatedParticles = (Body*)malloc(sizeof(Body) * relatedBodies);

    // Simulation iterations
    // BUG: if mpi-nbody.c get ran without arguments nIters equals 0.
    int nIters = (argv[2] != NULL) ? atoi(argv[2]) : 10;

    // ! Temporary workaround
    if (nIters == 0) {
        nIters = 10;
    }

    if (rank == MASTER) {
        // Master core will print an info block
        printf("\n----- n-body simulation -----\n");
        printf("#bodies:\t%d\n", nBodies);
        printf("#iterations:\t%d\n", nIters);
        printf("-----------------------------\n\n");
    }

    // Start computing
    for (int iter = 0; iter < nIters; iter++) {
        // Sync all cores before starting
        MPI_Barrier(MPI_COMM_WORLD);
        double iterStart = MPI_Wtime(); // Retrieve starting time
        const float dt = 0.01f; // Time step
        int relatedIndex = 0;

        for (int n = 0; n < numtasks; n++) {
            if (n == rank) {
                // If the core n is the root then send own particles to all cores
                MPI_Ibcast(workBuf, sendcount[rank], bodytype, rank, MPI_COMM_WORLD, &requests[rank]);
            } else {
                // If the core n isn't the root then receive related particles
                // from other cores.
                // Retrieve the offset to start storing related particles
                // in order to avoid overwrite
                int relatedOffset = relatedIndex * sendcount[n-1];
                if (n == 0) {
                    relatedOffset = 0;
                }
                // Receive related particles from rank n
                MPI_Ibcast(&relatedParticles[relatedOffset], sendcount[n], bodytype, n, MPI_COMM_WORLD, &requests[n]);
                // Increment relatedIndex in order to calculate different offsets
                relatedIndex++;
            }
        }
        
        // Start computing body force for own particles
        bodyForce(workBuf, dt, sendcount[rank]);

        // Catch all the request sent from other cores and compute
        for (int requestsCount = 0; requestsCount < numtasks - 1; requestsCount++) {
            int reqIndex;
            MPI_Waitany(numtasks, requests, &reqIndex, &status);
            // Check if the request is not sent by the same core who is receiving data
            if (reqIndex != rank && reqIndex < numtasks) {
                int bodiescount = sendcount[reqIndex];
                if (reqIndex > rank) {
                    reqIndex += -1;
                }
                int startOffset = reqIndex * bodiescount;
                // Compute body force for own particles relating to others
                relatedBodyForce(workBuf, sendcount[rank], &relatedParticles[startOffset], bodiescount, dt);
            }
        }
        
        // Integrate position for own particles
        for (int i = 0 ; i < sendcount[rank]; i++) {
            workBuf[i].x += workBuf[i].vx*dt;
            workBuf[i].y += workBuf[i].vy*dt;
            workBuf[i].z += workBuf[i].vz*dt;
        }

        // Sync all cores to take iteration time
        MPI_Barrier(MPI_COMM_WORLD);
        double iterEnd = MPI_Wtime();
        if (rank == MASTER) {
            // Master will print a string that indicates the iteration completition
            printf("Iteration #%d completed in %f seconds.\n", iter + 1, iterEnd - iterStart);
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
        double simulationTime = end - start;
        printf("\nSimulation completed in %f seconds.\n", simulationTime);
        double avgTime = simulationTime/(double)(nIters);
        printf("Avg. iteration time: %f seconds\n", avgTime);
        printf("Writing results into bodies-dataset.json ...\n");
        fflush(stdout);

        // Create and open bodies-dataset.json
        FILE *dataset = fopen("bodies-dataset.json", "w+");
        // Begin json structure
        fprintf(dataset, "{");
        // Write simulation info
        fprintf(dataset, "\t\"info\": {\n");
        fprintf(dataset, "\t\t\"bodies\": %d,\n", nBodies);
        fprintf(dataset, "\t\t\"interations\": %d,\n", nIters);
        fprintf(dataset, "\t\t\"simulation-time\": %f,\n", simulationTime);
        fprintf(dataset, "\t\t\"avg-iteration-time\": %f\n", avgTime);
        fprintf(dataset, "\t},\n");
        // Write bodies data
        for (int i = 0; i < nBodies; i++) {
            fprintf(dataset, "\t\"body[%d]\": {\n", i + 1);
            fprintf(dataset, "\t\t\"x\": %f,\n", commBuf[i].x);
            fprintf(dataset, "\t\t\"y\": %f,\n", commBuf[i].y);
            fprintf(dataset, "\t\t\"z\": %f,\n", commBuf[i].z);
            fprintf(dataset, "\t\t\"vx\": %f,\n", commBuf[i].vx);
            fprintf(dataset, "\t\t\"vy\": %f,\n", commBuf[i].vy);
            fprintf(dataset, "\t\t\"vz\": %f\n", commBuf[i].vz);
            if (i != nBodies-1) {
                fprintf(dataset, "\t},\n");
            } else {
                fprintf(dataset, "\t}\n");
            }
        }
        // End json structure
        fprintf(dataset, "}\n");
        // Close the file
        fclose(dataset);

        printf("Report generated into bodies-dataset.json.\n");
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
