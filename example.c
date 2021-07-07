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

    printf("\nOptimising %d dimensional Rosenbrock function\n\n", num_dims);

    // Run the optimiser for a number of steps
    for (int i = 0; i < num_evals; i++)
    {
        // Request a new candidate for fitness evaluation
        int id = de_ask(opt, candidate);

        // Run the fitness eval and report it back to the optimiser
        float fitness = rosenbrock(num_dims, candidate);
        de_tell(opt, id, candidate, fitness);

        // Query the current best fitness score
        float best_fitness = de_best(opt, NULL);

        // Every 100k evals, print the best fitness
        if (i % 100000 == 0)
        {
            printf("Step %d fitness:\t\t%.10e\n", i, best_fitness);
        }
    }

    // Now we're done, we also query the best candidate solution and print it out
    float best_fitness = de_best(opt, candidate);

    printf("\nFound solution at:\n");
    for (int i = 0; i < num_dims; i++)
    {
        if (i % 10 == 0)
        {
            printf("\n    ");
        }
        printf("%f, ", candidate[i]);
    }
    printf("\n\n");

    // Cleanup and return
    free(candidate);
    de_deinit(opt);
    return 0;
}
