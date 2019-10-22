'''Contains functions for calculating various metrics'''

import torch

def x_cov(x_matrix, y_matrix):
    '''
        Calculates the XCov penalty similar to 'Discovering Hidden Factors of Variation
        in Deep Networks' by Cheung et al.:
        Basically we calculate the cross-covariance matrix, normalize by dividing by the
        standard deviations, then square all elements, sum them and divide by the number of elements
        in the cross-covariance matrix.

        Parameters:
            x_matrix, y_matrix: torch Tensors that we calculate XCov for.
                                The size in dimension 0 is the batch size.

        Returns:
            x_cov: 1x1 torch tensor containing the result.
    '''
    x_matrix = x_matrix.clone().squeeze()
    y_matrix = y_matrix.clone().squeeze()

    assert x_matrix.dim() == 2, 'x_matrix does not have 2 dimensions in x_cov()'
    assert y_matrix.dim() == 2, 'y_matrix does not have 2 dimensions in x_cov()'

    x_mean = torch.mean(x_matrix, 0, keepdim=True)
    y_mean = torch.mean(y_matrix, 0, keepdim=True)

    x_matrix = x_matrix - x_mean
    y_matrix = y_matrix - y_mean

    stdev_x = torch.sqrt(torch.sum(torch.mul(x_matrix, x_matrix), 0) / (x_matrix.size(0) - 1))
    stdev_y = torch.sqrt(torch.sum(torch.mul(y_matrix, y_matrix), 0) / (y_matrix.size(0) - 1))

    # Use Bessel's Correction
    cross_cov_matrix = torch.matmul(x_matrix.t(), y_matrix) / (x_matrix.size(0) - 1)
    norm_cross_cov_matrix = torch.div(cross_cov_matrix, torch.ger(stdev_x, stdev_y))

    metric = torch.sum(torch.mul(norm_cross_cov_matrix, norm_cross_cov_matrix))

    return metric / (x_matrix.size(1) * y_matrix.size(1))

def average_variance(input_matrix):
    '''
        Computes the average of the individual variances for each component.
        This is equivalent to computing the mean of the diagonal of the
        covariance matrix of input_matrix.

        Parameters:
            input_matrix: torch.tensor, The rows contain examples of the vector, the columns
                          are individual components.

        Returns: the average variance.
    '''
    input_matrix = input_matrix.clone().squeeze()

    assert input_matrix.dim() == 2, 'input_matrix does not have 2 dimensions in average_variance()'

    return torch.mean(torch.var(input_matrix, 0))

def compute_2AFC(ground_truths, predictions):
    '''
        Calculates the 2AFC metric as described in the paper introducing the
        FERA17 challenge.

        Parameters:
            ground_truths: batch_size x n_components tensor of booleans,
                    describing the ground truths.
            predictions: batch_size x n_components tensor of float predictions
                    between 0 and 1, describing the predictions by the network

        Returns:
            component_2afc: tensor of length n_components containing the 2AFC values
                    for the individual components
            average_2afc: 1x1 tensor of the overall 2AFC value
    '''

    component_sum_values = torch.zeros_like(predictions[0])
    total_sum_value = torch.zeros(1)

    n_positive_gts = torch.sum(ground_truths, dim=0, dtype=torch.float)
    n_negative_gts = (torch.ones_like(n_positive_gts) * ground_truths.size()[0] - n_positive_gts)\
                    .to(dtype=torch.float)
    n_positive_gts += torch.finfo(torch.float).tiny
    n_negative_gts += torch.finfo(torch.float).tiny

    # Calculate 2AFC score for components
    # Because I expect n_negative_gts >> n_positive_gts, this implementation should
    # be slightly faster than vectorising and summing over all pairs
    for component_ind in range(ground_truths.size()[1]):
        positive_gt_inds = torch.nonzero(ground_truths[:, component_ind])
        negative_gt_inds = torch.nonzero(ground_truths[:, component_ind] == 0)

        for pos_ind in positive_gt_inds:
            for neg_ind in negative_gt_inds:
                if predictions[pos_ind, component_ind] > predictions[neg_ind, component_ind]:
                    component_sum_values[component_ind] += 1
                elif predictions[pos_ind, component_ind] == predictions[neg_ind, component_ind]:
                    component_sum_values[component_ind] += 0.5
                # else: add zero

    component_2afc = component_sum_values / (n_positive_gts * n_negative_gts)

    # Calculate overall 2AFC score
    all_positive_gt_inds = torch.nonzero(ground_truths)
    all_negative_gt_inds = torch.nonzero(ground_truths == 0)

    for pos_ind in all_positive_gt_inds:
        for neg_ind in all_negative_gt_inds:
            if predictions[pos_ind[0], pos_ind[1]] > predictions[neg_ind[0], neg_ind[1]]:
                total_sum_value += 1
            elif predictions[pos_ind[0], pos_ind[1]] == predictions[neg_ind[0], neg_ind[1]]:
                total_sum_value += 0.5
            # else: add zero

    average_2afc = total_sum_value / (torch.sum(n_positive_gts) * torch.sum(n_negative_gts))

    return component_2afc, average_2afc
