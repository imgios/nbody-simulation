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

// Body force calculation
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

// Randomize body position and velocity data
void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

int main (int argc, int ** argv) {
    int numtasks, rank, tag = 1;
    // Temporary initialization with fixed data
    int nBodies = 30000;
    int bytes = nBodies * sizeof(Body);
    float *buf = (float*)malloc(bytes);
    float *commBuf = (float*)malloc(bytes); // TODO: memory leak? Anyway to fix, this is probably too big
    //Body *p = (Body*)buf;

    MPI_Status status;
    MPI_Datatype bodyType;

    // Init MPI env
    MPI_Init(&argc, &argv);
    MPI_Type_contiguous( 6, MPI_FLOAT, &bodyType); // REVIEW: Contiguous or Struct?
    MPI_Type_commit(&bodyType);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) { // master
        // Init bodies position and velocity data
        randomizeBodies(buf, 6*nBodies);

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

    free(buf);
    MPI_Finalize();
}