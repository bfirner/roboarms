"""
This python file contains function to perform embedded space perturbations on robot arm data.
Perturbations can be thought of has two steps:
    1. First, choose a new location for the perturbation. This should be based upon a real data
    point and will occur within some distance of the actual data point. A new training target is
    also chosen, again based upon the original data point.
    2. Perturb the embedded space to match the perturbed location. The perturbation will be based
    upon the vector embedding of the original data and should have a noise component to reduce
    overfitting to errors in the perturbation.
"""

import math
import random
import torch


def similarSlope(slope):
    """Convenience function that quickly returns a new slope from a reasonable range."""
    # Usually return something that doesn't 0 out the slope, but also include some random stuff at
    # times. Slopes are normalized later, so don't worry about the ranges.
    # This was tuned by looking at some values on a graph, the numbers aren't necessarily
    # meaningful.
    return random.choice([random.uniform(-2*slope, -slope/20), random.uniform(slope/20, 2*slope),
        random.uniform(-1, -0.001), random.uniform(0.001, 1)])


def generatePerturbedXYZ(begin_xyz, end_xyz, step_size):
    """Using the vector from begin_xyz to end_xyz, choose a point perpendicular to that vector.

    """
    # If there is no movement requested then there is nothing to do
    if 0. == step_size:
        return begin_xyz
    x_slope = end_xyz[0] - begin_xyz[0]
    y_slope = end_xyz[1] - begin_xyz[1]
    z_slope = end_xyz[2] - begin_xyz[2]

    # Choose a random x and y slopes for the perturbed point that are generally consistent with the
    # existing movement in the x and y direction
    # TODO Consistent in the x and y axis is task specific
    p_x_slope = similarSlope(x_slope)
    p_y_slope = similarSlope(y_slope)
    # The z slope will be dependent upon the first two and will blow up if the original z slope was
    # 0. If the x, y, and z slopes are of the same order of magnitude then the z slope will be in
    # the same range.

    # Handle the case where the original z slope or any of the other slopes were zero. If this
    # happens then any non-zero slope with be orthogonal, but we will need to constrain the y slope.
    if 0. == z_slope and 0. == y_slope and 0. == x_slope:
        # In this case we can just choose any random numbers for all of the slopes
        p_x_slope = random.uniform(-step_size/3., step_size/3.)
        p_y_slope = random.uniform(-step_size/3., step_size/3.)
        p_z_slope = random.uniform(-step_size/3., step_size/3.)
    elif 0. == z_slope and 0. == y_slope:
        # In this case the y and z slopes can be random, but the x is constrained
        p_y_slope = random.uniform(-step_size/3., step_size/3.)
        p_z_slope = random.uniform(-step_size/3., step_size/3.)
        p_x_slope = -(y_slope*p_y_slope + z_slope*p_z_slope)/x_slope
    elif 0. == z_slope:
        # In this case we just need to randomize a z slope and then set the y from the x and z
        # slopes
        p_z_slope = random.uniform(-step_size/3., step_size/3.)
        p_y_slope = -(x_slope*p_x_slope + z_slope*p_z_slope)/y_slope
    else:
        p_z_slope = -(x_slope*p_x_slope + y_slope*p_y_slope)/z_slope

    # There is a degenerate case where all of the slopes are 0, as the null set is orthogonal to
    # every possible slope. Mathematically this should happen as it is impossible to land on exactly
    # 0 with a random number. Practically, the chance of getting two 0's from the uniform RNG is
    # vanishingly small and will only affect a single generated sample anyway.

    # Find a scale to take a step of the requested distance
    slope_distance = math.sqrt(p_x_slope**2 + p_y_slope**2 + p_z_slope**2)
    slope_scale = step_size / slope_distance
    
    # Generate the new starting point.
    pert_x = begin_xyz[0] + p_x_slope * slope_scale
    pert_y = begin_xyz[1] + p_y_slope * slope_scale
    pert_z = begin_xyz[2] + p_z_slope * slope_scale

    return (pert_x, pert_y, pert_z)


def findMultivariateParameters(sample_targets, correlations, feature_means, target_means):
    """Return the parameters of a multivariate distribution conditioned on the values of the sample targets.

    Arguments:
        sample_targets (torch.tensor): Values of the sample targets to use for conditioning.
        correlations   (torch.tensor): Matrix from torch.corfcoef.
                                       Early indices are for features, later are for targets.
        feature_means  (torch.tensor): Means of the features in the dataset.
        target_means   (torch.tensor): Means of the targets in the dataset.
    Returns:
        (mean_vector, covariance_vector)
    """
    num_targets = sample_targets.size(0)
    sigma11 = correlations[0:-num_targets, 0:-num_targets]
    sigma22 = correlations[-num_targets:, -num_targets:]
    sigma12 = correlations[0:-num_targets, -num_targets:]
    sigma21 = correlations[-num_targets:, 0:-num_targets]

    # The conditioned means of the features scaled with the distance of the targets from their means
    # and the covariance of the features with the conditioned targets. For example, if the
    # covariance value is 0 for a feature then it will not change at all, and if the covariance is 1
    # then the change will scale directly with the distance of the targets from their means.
    mean_vector = feature_means + torch.matmul(torch.matmul(sigma12, torch.linalg.inv(sigma22)), (sample_targets - target_means))
    # If the covariance is 1, then the variance of a feature would be 0 if the features are fixed.
    # Conversely, if the covariance is 0 then the variance of the feature remains unchanged.
    #cov_matrix = sigma11 - torch.matmul(torch.matmul(sigma12, torch.linalg.inv(sigma22)), sigma21)
    cov_matrix = sigma11 - torch.matmul(sigma12, torch.linalg.solve(sigma22, sigma21))
    # TODO FIXME sigma11 is not positive definite, so it cannot be used in the multivariate normal
    # distribution.

    return (mean_vector, cov_matrix)


def drawNewFeatures(mean_vector, cov_matrix, num_draws):
    """The the specified number of variables from a multivariate distrubution."""
    tx = torch.distributions.transforms.LowerCholeskyTransform()
    ltm = tx(cov_matrix)
    dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean_vector,
        scale_tril=ltm)
        #covariance_matrix=cov_matrix)
    samples = []
    for _ in range(num_draws):
        # Add the sample with a new batch dimension
        samples.append(dist.sample().expand(1, -1))
    # Concat samples along the batch dimension
    return torch.cat(samples, dim=0)


def makeSpherePoints(point_means, sample_points, start_radius):
    """Takes the point means and the sample points and creates new starting points for the samples.

    The begin points will lie on a sphere of the given radius from the mean locations and will lie
    along a straight vector from the center of the sphere to one of the sample points.

    Arguments:
        point_means   torch.tensor([[x,y,z]])
        sample_points torch.tensor([[x,y,z]])
        start_radius  float
    Returns:
        begin_points
    """
    distances = torch.pow(sample_points - point_means, 2).sum(dim=1, keepdim=False).sqrt()
    slopes = (sample_points - point_means)/distances

    begin_points = slopes * start_radius + point_means

    return begin_points


def perturbEmbedding(features, perturbation, correlations, slopes, noise_profiles):
    """Perturb the given features to the given perturbation.

    Arguments:
        features: A flat feature vector representing a DNN's feature embedding.
        perturbation: A list of new values for the features, or None for unchanging values
        correlation: A correlation matrix, from torch.corrcoef. Caller should zero out correlations
                     for any of the features being perturbed.
        noise_profiles: Noise profiles for each individual feature. This should have a value for all
                        features that are not given a value in perturbation.
    Returns:
        a new embedding with the same size as features
    """
    additions = torch.zeros(features.size())
    for i, p in enumerate(perturbation):
        # The delta between the 
        delta = p - features[i].item()
        additions[i] = delta
        # Assume independence between the perturbed features
        # Note that this may be a terribly wrong assumption
        for c_idx in range(correlations.size(1)):
            # TODO FIXME The correlation is normalized (to the range -1 to 1) and needs to be
            # denormalized with the slope to be used like this.
            additions[i] += math.abs(correlations[i][c_idx].item()) * delta * slope[i][c_idx]

    for j, noise in enumerate(noise_profiles):
        additions += random.normalvariate(noise.mean, noise.stddev)
    return features + additions

