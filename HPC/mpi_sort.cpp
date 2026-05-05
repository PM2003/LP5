#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
using namespace std;

// Merge two sorted arrays
vector<int> mergeArrays(const vector<int>& a, const vector<int>& b) {
    vector<int> result;
    int i = 0, j = 0;

    while (i < a.size() && j < b.size()) {
        if (a[i] <= b[j])
            result.push_back(a[i++]);
        else
            result.push_back(b[j++]);
    }

    while (i < a.size()) result.push_back(a[i++]);
    while (j < b.size()) result.push_back(b[j++]);

    return result;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 100000;  // total elements
    int local_n = N / size;

    vector<int> data;
    vector<int> local_data(local_n);

    double start, end;

    if (rank == 0) {
        data.resize(N);
        for (int i = 0; i < N; i++)
            data[i] = rand() % 1000000;

        start = MPI_Wtime();
    }

    // Scatter data
    MPI_Scatter(data.data(), local_n, MPI_INT,
                local_data.data(), local_n, MPI_INT,
                0, MPI_COMM_WORLD);

    // Local sort
    sort(local_data.begin(), local_data.end());

    vector<int> gathered;
    if (rank == 0)
        gathered.resize(N);

    // Gather sorted chunks
    MPI_Gather(local_data.data(), local_n, MPI_INT,
               gathered.data(), local_n, MPI_INT,
               0, MPI_COMM_WORLD);

    // Merge step (only root)
    if (rank == 0) {
        vector<int> sorted(local_n);
        copy(gathered.begin(), gathered.begin() + local_n, sorted.begin());

        for (int i = 1; i < size; i++) {
            vector<int> temp(gathered.begin() + i * local_n,
                             gathered.begin() + (i + 1) * local_n);
            sorted = mergeArrays(sorted, temp);
        }

        end = MPI_Wtime();
        cout << "Processes: " << size
             << " | Time: " << end - start << " sec" << endl;
    }

    MPI_Finalize();
    return 0;
}