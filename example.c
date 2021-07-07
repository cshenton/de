#include <stdio.h>

#define DIFFERENTIAL_EVOLUTION_IMPL
#include "de.h"

// Rosenbrock function, reasonably tricky non-convex function in n dimensions.
float rosenbrock(const int params_count, float *params)
{
    float delta = 0.0;
    float sum = 0.0;
    for (int i = 0; i < params_count - 1; i++)
    {
        delta = params[i + 1] - params[i] * params[i];
        sum += 100.0 * delta * delta;
        delta = 1.0 - params[i];
        sum += delta * delta;
    }

    return sum;
}

int main()
{
    const int num_evals = 3 * 1000 * 1000;
    const int num_dims = 50;

    // Initialise an optimiser for the 50 dimension rosenbrock function
    de_optimiser *opt = de_init(&(de_settings){
        .dimension_count = num_dims,
        .population_count = 100,
        .lower_bound = -2.0f,
        .upper_bound = 2.0f,
        .random_seed = 42,
    });

    // This is small enough to stack alloc, but we heap alloc for demonstration
    float *candidate = (float *)malloc(sizeof(float) * num_dims);

    // Early exit if de_init or candidate allocs failed
    if (!opt || !candidate)
    {
        printf("Out of memory\n");
        return 1;
    }

    // Run the optimiser for a number of steps
    for (int i = 0; i < num_evals; i++)
    {
        int id = de_ask(opt, candidate);

        // We want to minimise this
        float fitness = rosenbrock(num_dims, candidate);
        de_tell(opt, id, candidate, fitness);

        // Querying the best fitness without the candidate is very cheap
        float best_fitness = de_best(opt, NULL);
        if (i % 1000 == 0)
        {
            printf("Best fitness: %.10e\n", best_fitness);
        }
    }

    // Now we're done, we also query the best candidate solution
    float best_fitness = de_best(opt, candidate);

    // Cleanup and return
    free(candidate);
    de_deinit(opt);
    return 0;
}
