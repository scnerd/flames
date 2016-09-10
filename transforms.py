from variations import *

class Transform():
    '''
    Transforms are probabilistic collections of variations, with some extra fluff
    When a transform is called with x, y, it selects a random variation and runs it for that instance
    This can also integrate symmetry, including handling selection weighting to ensure that
    symmetry is properly represented in the final result
    Note that when a transform is given to the Chaos Game, it's actually unrolled, so
    each variation inside of it is handled separately (selection probabilities are retained)
    '''
    def __init__(self, *variations, weights=None, rotational_sym=1, dihedral_sym=False):
        n = len(variations)
        if n == 0:
            raise RuntimeError("Transform must have at least one variation")

        if weights is None:
            weights = np.random.uniform(size=len(variations))
        elif weights == 1:
            weights = np.ones(len(variations))
        else:
            if not len(weights) == len(variations):
                raise RuntimeError("Number of variations and weights for a Transform must be equal")

        weights = list(weights)
        variations = list(variations)

        total_weight = np.sum(weights)
        if dihedral_sym:
            variations.append(dihedral())
            weights.append(total_weight)
            total_weight *= 2

        for rot in range(rotational_sym-1):
            theta = 2 * np.pi * (rot + 1) / rotational_sym
            variations.append(rotate(theta))
            weights.append(total_weight)

        self.variations = variations
        self.weights = weights / np.sum(weights)

        print("Initialized transform: %s" % ", ".join(str(var) for var in self.variations))

    def __str__(self):
        return "Transform<%s>" % (";".join(str(var) for var in self.variations))

    def __len__(self):
        return len(self.variations)

    def weights(self):
        return self.weights

    def __iter__(self):
        return iter([(trans, w1 * w2) for var, w1 in zip(self.variations, self.weights) for trans, w2 in var])

def random_transform():
    num_trans = np.random.choice([1,2,3])
    transes = []
    for t in range(num_trans):
        num_vars = np.random.randint(3,8)
        variations = np.random.choice(list(all_variations), size=num_vars, replace=True)
        variations = [var() for var in variations]
        sym = np.random.choice([0, 1, 2, 2])
        if sym == 0:
            dih = False
            rot = 1
        elif sym == 1:
            dih = True
            rot = 1
        elif sym == 2:
            dih = True
            rot = np.random.choice([2,3,4,5])
        transes.append(Transform(*variations, rotational_sym=rot, dihedral_sym=dih))
    return transes

