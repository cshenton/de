# de

Single file Apache 2.0 licensed implementation of the Adaptive Differential Evolution optimiser.

Adaptive DE with radius limiting is among the state of the art in gradient-free optimisers. 
Gradient-free optimisers are theoretically very portable (since they require no information
from the problem other than fitness scores). However, I was not happy with the portability of
existing DE implementations.

Since this implementation is a single C99 header, it can be easily integrated with any existing
build system which supports compiling C code, and the API itself couldn't be simpler.

## Usage

Below is a quick tour of every interface in `de.h`. For a more complete example with error handling
and a real world optimisation problem, see `example.c`.

To use `de.h` include it where it is needed, and in _one file_ define `DIFFERENTIAL_EVOLUTION_IMPL`
before including the header.

```c
#define DIFFERENTIAL_EVOLUTION_IMPL
#include "de.h"
```

To initialise the library, fill out a `de_settings` struct with your problem dimensions, then call
`de_init`. If a memory allocation fails, it will return a null pointer.

```c
de_optimiser *opt = de_init(&(de_settings){
    .dimension_count = 50,      // Number of dimensions in the optimisation problem
    .population_count = 100,    // Number of agents in the population
    .lower_bound = -2.0f,       // Lower bound of the search space (same in all dimensions)
    .upper_bound = 2.0f,        // Upper bound of the search space (same in all dimensions)
    .random_seed = 42,          // Seed for the optimiser's RNG
});
```

Then (likely in a loop), call `de_ask` to retrieve an optimisation candidate. The library will also
return an id which you must hold onto.

```c
float candidate[50]; // For larger problems, we'll need to heap allocate this
int id = de_ask(opt, candidate);
```

Compute this candidate's fitness, then report it back to the optimiser with a call to `de_tell`.

```c
float fitness = my_fitness(candidate); // We want to minimise this
de_tell(opt, id, candidate, fitness);
```

At any point, you can query the optimiser for the best candidate solution and its corresponding
fitness value.

```c
float best_candidate[50];
float best_fitness = de_best(opt, best_candidate);
```

Finally, when you are finished using the optimiser, remember to free its memory with `de_deinit`.

```c
de_deinit(opt);
```

That's it! That's the entire library. See `example.c` for a complete example optimising the
rosenbrock function.
