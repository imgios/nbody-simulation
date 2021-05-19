#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f

// Struct for Bodies
typedef struct {
    float x, y, z; // position
    float vx, vy, vz; // velocity
} Body;

/*
* This function calculates the body force of each element of the array.
*
*   p: array of bodies
*   dt: delta
*   n: number of bodies
*/
void bodyForce (Body *p, float dt, int n) {
    for (int i = 0; i < n; i++) {
        float Fx = 0.0f;
        float Fy = 0.0f;
        float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = (dx * dx) + (dy * dy) + (dz * dz) + SOFTENING; // using parenthesis just for reading it more easily
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
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
    int bytes = nBodies * sizeof(Body);
    float *buf = (float*)malloc(bytes);
    float *commBuf = (float*)malloc(bytes); // TODO: memory leak? Anyway to fix, this is probably too big
    //Body *p = (Body*)buf;

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
    // TODO: Time calculation


    if (rank == 0) { // master
        // Init bodies position and velocity data
        randomizeBodies(buf, 6*nBodies);
        int sendcount[numtasks]; // Contains how many items every rank should receive

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

        int displacements[numtasks];
        displacements[0] = 0; // Master starts from index 0
        for (int i = 1; i < numtasks; i++) {
            // Init the displacement using the number of items sent and the index used
            // by the previous core.
            displacements[i] = sendcount[i-1] + displacements[i-1];
        }

        // TODO: Send datas to slaves
    } else { // slaves
        // TODO: Receive data from master
        const int nIters = 10; // Simulation iterations
        const float dt = 0.01f; // Time step
        Body *particles = (Body*)commBuf;
        int bodycount; // Number of bodies received
        for (int iter = 1; iter <= nIters; iter++) {
            bodyForce(particles, dt, nBodies); // Compute interbody forces

            for (int i = 0; i < nBodies; i++) { // Integrate position
                particles[i].x += p[i].vx * dt;
                particles[i].y += p[i].vy * dt;
                particles[i].z += p[i].vz * dt;
            }
        }
        free(commBuf);
    }

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all cores
    // TODO: Time calculation
    free(buf);
    MPI_Finalize();
}