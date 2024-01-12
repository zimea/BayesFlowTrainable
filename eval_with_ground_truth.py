def plot_posterior_2d(
    posterior_draws,
    prior=None,
    prior_draws=None,
    param_names=None,
    height=3,
    label_fontsize=14,
    legend_fontsize=16,
    tick_fontsize=12,
    post_color="#8f2727",
    prior_color="gray",
    post_alpha=0.9,
    prior_alpha=0.7,
    ground_truth=None,
    ground_truth_size=200,
    ground_truth_color='black',
    ground_truth_alpha=0.9,
    ground_truth_shape='+'
):
    """Generates a bivariate pairplot given posterior draws and optional prior or prior draws.

    posterior_draws   : np.ndarray of shape (n_post_draws, n_params)
        The posterior draws obtained for a SINGLE observed data set.
    prior             : bayesflow.forward_inference.Prior instance or None, optional, default: None
        The optional prior object having an input-output signature as given by ayesflow.forward_inference.Prior
    prior_draws       : np.ndarray of shape (n_prior_draws, n_params) or None, optonal (default: None)
        The optional prior draws obtained from the prior. If both prior and prior_draws are provided, prior_draws
        will be used.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    height            : float, optional, default: 3
        The height of the pairplot
    label_fontsize    : int, optional, default: 14
        The font size of the x and y-label texts (parameter names)
    legend_fontsize   : int, optional, default: 16
        The font size of the legend text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    post_color        : str, optional, default: '#8f2727'
        The color for the posterior histograms and KDEs
    priors_color      : str, optional, default: gray
        The color for the optional prior histograms and KDEs
    post_alpha        : float in [0, 1], optonal, default: 0.9
        The opacity of the posterior plots
    prior_alpha       : float in [0, 1], optonal, default: 0.7
        The opacity of the prior plots
    ground_truth : np.array of shape (n_params) or None, optional, default: None
        The ground truth parameter set used to generate the simulation
    ground_truth_size : size of the ground_truth points
    ground_truth_color: color of the ground_truth points
    ground_truth_alpha: opacity of the ground_truth points
    ground_truth_shape: shape of the ground_truth points

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    AssertionError
        If the shape of posterior_draws is not 2-dimensional.
    """

    # Ensure correct shape
    assert (
        len(posterior_draws.shape)
    ) == 2, "Shape of `posterior_samples` for a single data set should be 2 dimensional!"

    # Obtain n_draws and n_params
    n_draws, n_params = posterior_draws.shape

    # If prior object is given and no draws, obtain draws
    if prior is not None and prior_draws is None:
        draws = prior(n_draws)
        if type(draws) is dict:
            prior_draws = draws["prior_draws"]
        else:
            prior_draws = draws
    # Otherwise, keep as is (prior_draws either filled or None)
    else:
        pass

    # Attempt to determine parameter names
    if param_names is None:
        if hasattr(prior, "param_names"):
            if prior.param_names is not None:
                param_names = prior.param_names
            else:
                param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]
        else:
            param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Pack posterior draws into a dataframe
    posterior_draws_df = pd.DataFrame(posterior_draws, columns=param_names)

    # Add posterior
    g = sns.PairGrid(posterior_draws_df, height=height)
    g.map_diag(sns.histplot, fill=True, color=post_color, alpha=post_alpha, kde=True)
    g.map_lower(sns.kdeplot, fill=True, color=post_color, alpha=post_alpha)

    # Add prior, if given
    if prior_draws is not None:
        prior_draws_df = pd.DataFrame(prior_draws, columns=param_names)
        g.data = prior_draws_df
        g.map_diag(sns.histplot, fill=True, color=prior_color, alpha=prior_alpha, kde=True, zorder=-1)
        g.map_lower(sns.kdeplot, fill=True, color=prior_color, alpha=prior_alpha, zorder=-1)

    # Add legend, if prior also given
    if prior_draws is not None or prior is not None:
        handles = [
            Line2D(xdata=[], ydata=[], color=post_color, lw=3, alpha=post_alpha),
            Line2D(xdata=[], ydata=[], color=prior_color, lw=3, alpha=prior_alpha),
        ]
        legend_entries = ["Posterior", "Prior"]

    if ground_truth is not None:
        g.data = pd.DataFrame(ground_truth, columns=param_names)
        g.map_lower(plt.scatter, color=ground_truth_color, marker=ground_truth_shape, s=ground_truth_size, label='Ground truth')
        if handles:
            handles.append(Line2D(xdata=[], ydata=[], color=ground_truth_color, lw=3, alpha=ground_truth_alpha))
            legend_entries.append("Ground truth")
        else:
            handles = [
                Line2D(xdata=[], ydata=[], color=post_color, lw=3, alpha=post_alpha),
                Line2D(xdata=[], ydata=[], color=ground_truth_color, lw=3, alpha=ground_truth_alpha),
            ]
        legend_entries = ["Posterior", "Ground truth"]

    if handles:
        g.fig.legend(handles, legend_entries, fontsize=legend_fontsize, loc="center right")

    # Remove upper axis
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[i, j].axis("off")

    # Modify tick sizes
    for i, j in zip(*np.tril_indices_from(g.axes, 1)):
        g.axes[i, j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
        g.axes[i, j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Add nice labels
    for i, param_name in enumerate(param_names):
        g.axes[i, 0].set_ylabel(param_name, fontsize=label_fontsize)
        g.axes[len(param_names) - 1, i].set_xlabel(param_name, fontsize=label_fontsize)

    # Add grids
    for i in range(n_params):
        for j in range(n_params):
            g.axes[i, j].grid(alpha=0.5)

    g.tight_layout()
    return g.fig