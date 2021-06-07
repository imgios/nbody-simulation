<!-- final project report -->
| Studente | MD5 | Progetto|
|:--------:|:----|:-------:|
| Giosuè Sulipano | GIOSUÈSULIPANO → `f6a0b03e4847f5b0435e16d22daca582` | `n-body` - `m4.large` |

# Tabella dei contenuti
- [Introduzione](#introduzione)
- [Soluzione](#soluzione)
  * [Dettagli implementativi](#dettagli-implementativi)
    + [Inizializzazione](#inizializzazione)
    + [Computazione](#computazione)
    + [Terminazione](#terminazione)
- [Compilazione ed Esecuzione](#compilazione-ed-esecuzione)
- [Benchmarks](#benchmarks)
  * [Scalabilità debole](#scalabilit--debole)
  * [Scalabilità forte](#scalabilit--forte)
  * [Risultati](#risultati)

# Introduzione
**N-body** è un problema in cui bisogna predire i movimenti individuali di un gruppo di oggetti celesti che interagiscono tra loro. Tale problema lo si ritrova in diverse discipline, ad esempio (1) in astrofisica, in cui un astrofisico vorrebbe conoscere le posizioni e le velocità di un gruppo di stelle, (2) o nella chimica, in cui un chimico vorrebbe conoscere le posizioni e le velocità di una collezione di molecole o atomi.

Per la sua risoluzione viene considerato l'utilizzo di MPI per simulare il comportamento di un insieme di particelle tramite un algoritmo parallelo, cercando di mitigare l'overhead delle comunicazioni.

# Soluzione
La soluzione che viene proposta, basata su quella fornita dall'utente [harrism](https://github.com/harrism/mini-nbody/blob/master/nbody.c), fa utilizzo di MPI per simulare in modo parallelo il comportamento di un insieme di particelle.

Questa è ovviamente quadratica in quanto, per calcolare le posizioni delle particelle in ogni instati di tempo, ognuna di queste viene confrontata con tutte le altre particelle.

## Dettagli implementativi
La soluzione può essere vista come l'insieme di tre blocchi:
- Inizializzazione: in questa prima parte viene inizializzato MPI e tutte le risorse necessarie per la computazione, con una prima comunicazione delle particelle che ogni core deve computare;
- Computazione: in questa seconda parte vi è la vera e propria computazione della posizione delle particelle, confrontando ognuna di queste con le altre.
- Terminazione: in questa terza e ultima parte vi è l'export dei dati ottenuti, la terminazione di MPI, la pulizia della memoria allocata e la terminazione del programma.

### Inizializzazione
In questa fase del programma vengono definiti i buffer da utilizzare per la comunicazione e la computazione delle particelle, il tipo di dato derivato da utilizzare per le comunicazioni tra core diversi e viene inizializzato MPI.

Nella soluzione proposta si fa utilizzo di una Struct denominata `Body` e definita come l'insieme delle variabili che definiscono la posizione e la velocità delle particelle:
```c
typedef struct {
    float x, y, z; // position
    float vx, vy, vz; // velocity
} Body;
```
Per facilitare la comunicazione delle particelle tra core viene definito un nuovo tipo di dato derivato di tipo Struct denominato `bodytype`:
```c
MPI_Datatype bodytype, oldtypes[1];
MPI_Aint offsets[1];
int blockcounts[1];
offsets[0] = 0;
oldtypes[0] = MPI_FLOAT;
blockcounts[0] = 6;
MPI_Type_create_struct(1, blockcounts, offsets, oldtypes, &bodytype);
MPI_Type_commit(&bodytype);
```

Inoltre, il core MASTER (ovvero quello id 0) si occupa di:
- inizializzare le particelle attraverso un algoritmo randomico:
```c
void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}
```
- comunicare il numero di particelle che ogni core dovrà elaborare:
```c
...

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

...

// Send item counts to all cores
MPI_Bcast(sendcount, numtasks, MPI_INT, MASTER, MPI_COMM_WORLD);
```
- divide le particelle ai core attraverso una Scatterv:
```c
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
```
### Computazione
In questa fase avviene la vera e propria computazione delle posizioni delle particelle, facendo utilizzo di comunicazioni non bloccanti e di buffer secondari in cui poter ricevere le particelle degli altri core.

Più nel dettaglio, viene definito un buffer secondario di dimensioni ridotte in cui poter conservare le particelle degli altri core ricevute ad ogni iterazione:
```c
int relatedBodies = nBodies - sendcount[rank]; // total bodies - own bodies
Body *relatedParticles = (Body*)malloc(sizeof(Body) * relatedBodies);
```

Ad ogni iterazione, ogni core invia le proprie particelle e riceve le particelle degli altri core in modo non bloccante.
```c
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
```

Facendo utilizzo delle comunicazioni non bloccanti è possibile che i core entrino in uno stato di stallo. Per sfruttare questa situazione al meglio viene effettuato un primo lavoro di computazione sulle proprie particelle confrotandole soltanto tra loro. Successivamente, per ogni porzione di particelle ricevuta, viene effettuato un secondo lavoro di computazione sulle proprie particelle in relazione a quelle ricevute. Infine, viene effettato un calcolo per integrare la posizione delle proprie particelle.

```c
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
```
A tal proposito sono state previste due implementazioni dell'algoritmo bodyForce:
```c
/*
* This function calculates the body force of every element of the array.
*
*   p: array of bodies
*   dt: time step
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
```
La differenza tra i due algoritmi consiste nel fatto che il primo algoritmo prende in input un unico array ed effettua i calcoli su tutte le particelle in relazione tra loro, mentre il secondo prende in input due array di particelle ed effettua i calcoli sulle particelle del primo in relazione con quelle del secondo array.

### Terminazione
In quest'ultima fase, il core MASTER raccoglie tutte le particelle dagli altri core mediante una Gatherv:
```c
// Gather all particles to the Master's communication buffer
MPI_Gatherv(workBuf, sendcount[rank], bodytype, commBuf, sendcount, displacements, bodytype, MASTER, MPI_COMM_WORLD);
```
Inoltre, fonisce un recap della simulazione effettuata e si occupa dell'export dei dati in un file `.json`:
```c
...

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
```

Infine, ogni core effettua delle operazioni di pulizia e di terminazione della libreria MPI:
```c
MPI_Type_free(&bodytype);
free(workBuf);
if (rank == MASTER) {
    free(commBuf);
}

MPI_Finalize();
```
# Compilazione ed Esecuzione
Per la compilazione ed esecuzione del codice è necessario avere [`OpenMPI`](https://www.open-mpi.org/).

- Compilazione: `mpicc mpi-nbody.c -o nbody -lm`
- Esecuzione: `mpirun nbody nbodies niters`, dove `nbodies` corrisponde al numero di particelle e `niters` al numero di iterazioni. Di default, questi equivalgono corrispettivamente a `30000` e `10`.
  * `Read -1, expected ####, errno = 1` durante l'esecuzione in un container docker? Il problema sembra essere relativo al Vader e viene discusso [qui](https://github.com/open-mpi/ompi/issues/4948). Eventuali soluzioni:
    + `mpirun --mca btl ^vader nbody nbodies niters`
    + `mpirun --mca btl_vader_single_copy_mechanism none nbody niters`

Al termine dell'esecuzione sarà disponibile il risultato della simulazione all'interno del file `bodies-dataset.json`.

# Benchmarks
## Scalabilità debole
## Scalabilità forte
## Risultati
