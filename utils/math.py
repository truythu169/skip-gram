from scipy import linalg, stats


def cos(vec1, vec2):
    norm1 = linalg.norm(vec1)
    norm2 = linalg.norm(vec2)

    return vec1.dot(vec2) / (norm1 * norm2)

def rho(vec1,vec2):
    return stats.stats.spearmanr(vec1, vec2)[0]

