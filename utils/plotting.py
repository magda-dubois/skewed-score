import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import arviz as az
import seaborn as sns
from utils.config_plotting import PLOT_STYLE, GRADER_COLORS, GRADER_ALIASES, GRADER_SETS
from matplotlib.patches import Patch
import traceback
from matplotlib.colors import to_rgb


def plot_predictive_distributions(idata, fig_path=None):
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

    az.plot_ppc(idata, var_names=["obs"], group="prior", num_pp_samples=80, ax=axes[0])
    axes[0].set_title("Prior Predictive Distribution", fontsize=PLOT_STYLE["title_size"])
    axes[0].set_xlabel("Score", fontsize=PLOT_STYLE["label_size"])
    axes[0].tick_params(axis='both', labelsize=PLOT_STYLE["tick_size"])
    axes[0].legend(fontsize=PLOT_STYLE["legend_size"], borderpad=0.3, labelspacing=0.2, loc='best')

    az.plot_ppc(idata, var_names=["obs"], group="posterior", kind="kde", num_pp_samples=50, ax=axes[1])
    axes[1].set_title("Posterior Predictive Distribution", fontsize=PLOT_STYLE["title_size"])
    axes[1].set_xlabel("Score", fontsize=PLOT_STYLE["label_size"])
    axes[1].tick_params(axis='both', labelsize=PLOT_STYLE["tick_size"])
    axes[1].legend(fontsize=PLOT_STYLE["legend_size"], borderpad=0.3, labelspacing=0.2, loc='upper left')

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=PLOT_STYLE["dpi"], bbox_inches='tight')
    plt.show()


def get_palettes(scheme_name):
    """
    Generate box and strip palettes for the specified grader set.
    
    Args:
        scheme_name: Name of the grader set from GRADER_SETS
    
    Returns:
        Tuple of (box_palette, strip_palette) dictionaries
    """
    # Get list of graders for this scheme
    graders = GRADER_SETS[scheme_name]
    
    # Create palettes
    box_palette = {}
    strip_palette = {}
    
    for grader in graders:
        # Resolve alias if needed
        if grader in GRADER_ALIASES:
            color_key = GRADER_ALIASES[grader]
        else:
            color_key = grader
            
        # Get colors
        colors = GRADER_COLORS.get(color_key, ('#CCCCCC', '#666666'))  # Default 
            
        box_palette[grader] = colors[0]    # Light for boxplot
        strip_palette[grader] = colors[1]  # Dark for stripplot
            
    return box_palette, strip_palette


def plot_model_comparison(idata_dict, ylabel=None, fig_path=None):
    """
    Compare models using LOO and WAIC and plot side-by-side comparisons.

    Parameters:
    - idata_dict (dict): Dictionary with model names as keys and InferenceData as values.
    - ylabel (str, optional): Label for y-axis (shared across both plots).
    - fig_path (str, optional): If provided, saves the figure to this path.
    """
    # Compute comparisons
    loo_comparison = az.compare(idata_dict, ic="loo")
    waic_comparison = az.compare(idata_dict, ic="waic")

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 2 plots side by side

    # Plot LOO
    az.plot_compare(loo_comparison, ax=axes[0])
    axes[0].set_title("Model comparison (LOO)", fontsize=PLOT_STYLE["title_size"])
    axes[0].set_xlabel("Relative ELPD", fontsize=PLOT_STYLE["label_size"])
    axes[0].set_ylabel(ylabel if ylabel else "", fontsize=PLOT_STYLE["label_size"])
    axes[0].tick_params(axis='both', labelsize=PLOT_STYLE["tick_size"])

    # Plot WAIC
    az.plot_compare(waic_comparison, ax=axes[1])
    axes[1].set_title("Model comparison (WAIC)", fontsize=PLOT_STYLE["title_size"])
    axes[1].set_xlabel("Relative ELPD", fontsize=PLOT_STYLE["label_size"])
    axes[1].set_ylabel("", fontsize=PLOT_STYLE["label_size"])
    axes[1].tick_params(axis='both', labelsize=PLOT_STYLE["tick_size"])

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=PLOT_STYLE["dpi"], bbox_inches='tight')
    plt.show()



def plot_cutpoint_positions(idata, fig_path=None, title="Cutpoint positions on latent scale", fig_size=(8, 4)):
    """
    Creates a visualization of ordinal cutpoint positions on the latent scale.
    
    Args:
        idata: InferenceData object containing the posterior samples
        fig_path: Optional path to save the figure
        title: Plot title
    """
    # Extract posterior mean values
    first_cutpoint = idata.posterior.first_cutpoint.mean(dim=("chain", "draw")).values
    cutpoint_diffs = idata.posterior.cutpoint_diffs.mean(dim=("chain", "draw")).values

    # Calculate absolute positions of all cutpoints
    cutpoints = np.zeros(len(cutpoint_diffs) + 1)
    cutpoints[0] = first_cutpoint
    for i in range(1, len(cutpoints)):
        cutpoints[i] = cutpoints[i-1] + cutpoint_diffs[i-1]

    # Create figure
    plt.figure(figsize=fig_size)

    # Draw horizontal line representing latent scale
    plt.axhline(y=0, color='black', linewidth=2)

    # Plot cutpoints as circles on the line
    plt.scatter(cutpoints, np.zeros_like(cutpoints), s=100, color='blue', zorder=3)

    # Add cutpoint values below the line
    for i, cp in enumerate(cutpoints):
        plt.text(cp, -0.1, f"{cp:.2f}", ha='center', fontsize=PLOT_STYLE["tick_size"])
        plt.text(cp, 0.1, f"{i}|{i+1}", ha='center', fontsize=PLOT_STYLE["tick_size"])

    # Add score labels in the middle of each region
    for i in range(len(cutpoints) + 1):
        if i == 0:
            # First region (score 0)
            position = cutpoints[0] - 2
        elif i == len(cutpoints):
            # Last region (score 10)
            position = cutpoints[-1] + 2
        else:
            # Middle regions
            position = (cutpoints[i-1] + cutpoints[i]) / 2
        
        plt.text(position, 0, f"{i}", ha='center', va='center', 
                fontsize=PLOT_STYLE["legend_size"], 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    # Remove y-axis ticks and labels
    plt.yticks([])
    plt.ylabel('')

    # Set x-axis limits with padding
    min_x = cutpoints.min() - 3
    max_x = cutpoints.max() + 3
    plt.xlim(min_x, max_x)
    plt.ylim(-0.5, 0.5)

    # Apply styling
    plt.title(title, fontsize=PLOT_STYLE["title_size"])
    plt.xlabel('Latent scale', fontsize=PLOT_STYLE["label_size"])
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    # Save if path provided
    if fig_path:
        plt.savefig(fig_path, dpi=PLOT_STYLE["dpi"], bbox_inches='tight')
    plt.show()



def draw_forest_plot(ax, idata, var_names, labeller=None, xlim=[-2.5, 2.5], vertical_centering=False):
    forest_plot = az.plot_forest(
        idata,
        var_names=var_names,
        labeller=labeller,
        hdi_prob=0.95,
        combined=True,
        ax=ax,
        show=False
    )

    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlim(xlim)

    # Dynamically determine how many rows were plotted
    try:
        y_labels = ax.get_yticklabels()
        n_rows = len([label for label in y_labels if label.get_text().strip() != ""])
    except Exception:
        n_rows = len(var_names)

    if vertical_centering:
        ax.set_ylim(-0.5, n_rows - 0.5)

    tick_step = max(1, round((xlim[1] - xlim[0]) / 5))
    xticks = np.arange(xlim[0], xlim[1] + tick_step, tick_step)
    ax.set_xticks(xticks)

    ax.set_xlabel('Effect sizes', fontsize=PLOT_STYLE["label_size"])
    ax.set_ylabel('Effects', fontsize=PLOT_STYLE["label_size"])
    ax.set_title('Posterior parameter estimates', fontsize=PLOT_STYLE["title_size"])
    ax.tick_params(axis='both', labelsize=PLOT_STYLE["tick_size"])

    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1)

def add_predictions_to_plot(ax, df, idata, x_var, dodge, pred_alpha, category_maps):
    """Add model predictions as overlay on existing plot"""
    try:
        posterior_pred = idata.posterior_predictive['obs'].values[0]
        pred_mean = posterior_pred.mean(axis=0)
        pred_lower = np.percentile(posterior_pred, 5, axis=0)
        pred_upper = np.percentile(posterior_pred, 95, axis=0)
        
        # Match predictions to data
        df_with_pred = df.copy().reset_index(drop=True)
        df_with_pred['pred_mean'] = pred_mean[:len(df)]
        df_with_pred['pred_lower'] = pred_lower[:len(df)]
        df_with_pred['pred_upper'] = pred_upper[:len(df)]
        
        # Check if we're in items mode by looking for the combined x variable
        if 'x_grader_item' in df.columns:
            # Items mode: predict for each x_var + grader + item combination
            add_item_level_predictions(ax, df_with_pred, x_var, pred_alpha)
        else:
            # Regular mode: use existing logic
            add_regular_level_predictions(ax, df_with_pred, x_var, dodge, pred_alpha)
            
    except Exception as e:
        print(f"Warning: Could not add predictions to plot: {e}")
        traceback.print_exc()


def add_item_level_predictions(ax, df_with_pred, x_var, pred_alpha):
    """Add predictions when items are shown as separate boxplots"""
    
    # Get the structure from the data
    x_categories = sorted(df_with_pred[x_var].unique())
    graders = sorted(df_with_pred['grader'].unique())
    items = sorted(df_with_pred['item'].unique())
    
    print(f"Adding item-level predictions for: {x_categories} × {graders} × {items}")
    
    # Calculate predictions for each x_var + grader + item combination
    plot_position = 0
    for x_val in x_categories:
        for grader in graders:
            for item in items:
                mask = ((df_with_pred[x_var] == x_val) & 
                       (df_with_pred['grader'] == grader) & 
                       (df_with_pred['item'] == item))
                
                if mask.sum() > 0:
                    # Get group-level statistics
                    group_pred_mean = df_with_pred[mask]['pred_mean'].mean()
                    group_pred_lower = df_with_pred[mask]['pred_lower'].mean()
                    group_pred_upper = df_with_pred[mask]['pred_upper'].mean()
                    
                    actual_mean = df_with_pred[mask]['score'].mean()
                    print(f"Plotting: {x_val} + {grader} + Item{item} -> position {plot_position}: Actual={actual_mean:.1f}, Pred={group_pred_mean:.1f}")
                    
                    # Plot prediction point
                    ax.scatter(plot_position, group_pred_mean, marker='D', s=50, color='red', 
                              alpha=pred_alpha, zorder=10, edgecolors='darkred', linewidth=1)
                    
                    # Plot credible interval
                    ax.plot([plot_position, plot_position], [group_pred_lower, group_pred_upper], 
                           'r-', linewidth=2, alpha=pred_alpha*0.8, zorder=9)
                
                plot_position += 1


def add_regular_level_predictions(ax, df_with_pred, x_var, dodge, pred_alpha):
    """Add predictions for regular (non-item) plots - your existing logic"""
    
    # Get the ACTUAL x-axis and hue order from the plot
    x_tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    x_categories = x_tick_labels if x_tick_labels[0] != '' else sorted(df_with_pred[x_var].unique())
    
    # Get hue order from seaborn's legend
    legend = ax.get_legend()
    if legend:
        hue_order = [t.get_text() for t in legend.get_texts()]
    else:
        hue_order = sorted(df_with_pred['grader'].unique())
    
    if x_var == 'grader':
        # Simple case
        for grader in hue_order:
            mask = df_with_pred['grader'] == grader
            if mask.sum() > 0:
                group_pred_mean = df_with_pred[mask]['pred_mean'].mean()
                group_pred_lower = df_with_pred[mask]['pred_lower'].mean()
                group_pred_upper = df_with_pred[mask]['pred_upper'].mean()
                
                try:
                    x_pos = x_categories.index(grader)
                except ValueError:
                    continue
                
                ax.scatter(x_pos, group_pred_mean, marker='D', s=50, color='red', 
                          alpha=pred_alpha, zorder=10, edgecolors='darkred', linewidth=1)
                ax.plot([x_pos, x_pos], [group_pred_lower, group_pred_upper], 
                       'r-', linewidth=2, alpha=pred_alpha*0.8, zorder=9)
    else:
        # Complex case: Use seaborn's exact dodge calculation
        n_hue = len(hue_order)
        
        # Seaborn's dodge calculation (simplified)
        if dodge and n_hue > 1:
            # Width of each hue group (seaborn default)
            width = 0.8  # Default boxplot width
            # Calculate positions for each hue level
            positions = []
            for i in range(n_hue):
                # This mimics seaborn's internal calculation
                offset = (i - (n_hue - 1) / 2) * (width / n_hue)
                positions.append(offset)
            print(f"Calculated dodge positions: {positions}")
        else:
            positions = [0] * n_hue
        
        for x_idx, x_val in enumerate(x_categories):
            for hue_idx, grader in enumerate(hue_order):
                mask = (df_with_pred[x_var] == x_val) & (df_with_pred['grader'] == grader)
                if mask.sum() > 0:
                    group_pred_mean = df_with_pred[mask]['pred_mean'].mean()
                    group_pred_lower = df_with_pred[mask]['pred_lower'].mean()
                    group_pred_upper = df_with_pred[mask]['pred_upper'].mean()
                    
                    # Use seaborn's exact positioning
                    x_pos = x_idx + positions[hue_idx]
                    
                    actual_mean = df_with_pred[mask]['score'].mean()
                    print(f"Plotting: {x_val} + {grader} -> position {x_pos:.3f}: Pred={group_pred_mean:.1f}")
                    
                    ax.scatter(x_pos, group_pred_mean, marker='D', s=50, color='red', 
                              alpha=pred_alpha, zorder=10, edgecolors='darkred', linewidth=1)
                    ax.plot([x_pos, x_pos], [group_pred_lower, group_pred_upper], 
                           'r-', linewidth=2, alpha=pred_alpha*0.8, zorder=9)



def darken(color, amount=0.8):
    """Darken an RGB color by scaling brightness."""
    r, g, b = to_rgb(color)
    return (r * amount, g * amount, b * amount)


def draw_violin_plot(ax, df, x_var, color_scheme, x_label='Evaluated model',
                        dodge=False, y_lim=[-0.5, 11.5], legend_title=None,
                        show_predictions=False, idata=None, pred_alpha=0.7,
                        category_maps=None, show_items=False, jitter_strength=0.15, legend_fontsize = None):

    box_palette, strip_palette = get_palettes(color_scheme)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    # Show humans first (better esthetics)
    graders = sorted(df['grader'].unique(), key=lambda x: (0 if x.startswith('Human') else 1, x))

    # colours
    if isinstance(box_palette, dict):
        palette_list = [box_palette[key] for key in sorted(box_palette.keys())]
    else:
        palette_list = list(box_palette) if hasattr(box_palette, '__iter__') else [box_palette]

    color_map = {
        grader: palette_list[i % len(palette_list)]
        for i, grader in enumerate(graders)
    }

    if show_items and 'item' in df.columns:
        df = df.copy()
        df['x_grader_item'] = (
            df[x_var].astype(str) + '_' +
            df['grader'].astype(str) + '_' +
            df['item'].astype(str)
        )

        x_categories = sorted(df[x_var].unique())
        items = sorted(df['item'].unique())
        x_order = [
            f"{x_cat}_{grader}_{item}"
            for x_cat in x_categories
            for grader in graders
            for item in items
        ]

        df['color'] = df['grader'].map(color_map)
        formatted_title = f'Scores by graders and items (N={len(df)})'

        for xtick in x_order:
            subset = df[df['x_grader_item'] == xtick]
            if not subset.empty:
                box_color = subset['color'].iloc[0]
                dot_color = darken(box_color, 0.8)

                sns.violinplot(
                    x='x_grader_item', y='score', data=subset,
                    color=box_color, ax=ax,
                    width=PLOT_STYLE["box_width"],
                    order=[xtick],
                    cut=0,
                    bw_adjust=0.2,
                    density_norm='width',
                    linewidth=0.05
                )

                sns.stripplot(
                    x='x_grader_item', y='score', data=subset,
                    color=dot_color, ax=ax,
                    alpha=1.0,
                    jitter=PLOT_STYLE["jitter"],
                    size=PLOT_STYLE["point_size"],
                    order=[xtick],
                    legend=False
                )

        create_custom_grader_legend(ax, graders, palette_list, legend_title, legend_fontsize)
        customize_simple_xaxis(ax, x_categories, graders, items)

    else:
        formatted_title = f'Scores by graders (N={len(df)})'
        order = sorted(df[x_var].unique())

        hue_order = graders
        n_grader = len(hue_order)
        width = PLOT_STYLE["box_width"]
        dodge_amount = width / n_grader

        # Violinplot with hue/dodge
        sns.violinplot(
            x=x_var, y='score', hue='grader', data=df,
            palette=box_palette, ax=ax,
            width=width,
            dodge=True,
            inner=None,
            linewidth=0.1,
            order=order,
            hue_order=hue_order
        )

        # Manually aligned scatter
        for i, grader in enumerate(hue_order):
            sub_df = df[df['grader'] == grader]
            for j, category in enumerate(order):
                scores = sub_df[sub_df[x_var] == category]['score']
                if scores.empty:
                    continue
                base_x = j
                offset = (i - (n_grader - 1) / 2) * dodge_amount
                jitter = (np.random.rand(len(scores)) - 0.5) * jitter_strength if PLOT_STYLE["jitter"] else 0
                x_vals = np.full(len(scores), base_x + offset) + jitter
                ax.scatter(
                    x_vals,
                    scores,
                    color=strip_palette[grader],
                    alpha=PLOT_STYLE["alpha"],
                    s=PLOT_STYLE["point_size"],
                    label=None,
                    zorder=3
                )

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)

        # Legend for multiple graders
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                title=legend_title if legend_title else None,
                fontsize=legend_fontsize if legend_fontsize else PLOT_STYLE["legend_size"],
                title_fontsize=legend_fontsize if legend_fontsize else PLOT_STYLE["legend_size"],
                frameon=False,
                borderpad=PLOT_STYLE["legend_borderpad"],
                labelspacing=PLOT_STYLE["legend_labelspacing"],
                loc = 'upper right',
            )

    if show_predictions and idata is not None and category_maps is not None:
        add_predictions_to_plot(ax, df, idata, x_var, dodge, pred_alpha, category_maps)

    ax.set_title(formatted_title, fontsize=PLOT_STYLE["title_size"])
    ax.set_ylabel('Score', fontsize=PLOT_STYLE["label_size"])
    ax.set_xlabel(x_label, fontsize=PLOT_STYLE["label_size"])
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', labelsize=PLOT_STYLE["tick_size"])


def create_custom_grader_legend(ax, graders, palette_list, legend_title, legend_fontsize):
    """Create legend showing grader colors"""
    
    
    legend_elements = []
    for i, grader in enumerate(graders):
        color = palette_list[i % len(palette_list)]
        legend_elements.append(Patch(facecolor=color, label=grader))
    
    legend_font = FontProperties(size=legend_fontsize if legend_fontsize else PLOT_STYLE["legend_size"])
    legend = ax.legend(
        handles=legend_elements,
        title=legend_title,
        prop=legend_font,
        title_fontproperties=legend_font,
        loc='upper right',
        borderpad=PLOT_STYLE["legend_borderpad"],
        labelspacing=PLOT_STYLE["legend_labelspacing"]
    )
    legend.get_frame().set_linewidth(0.0)


def customize_simple_xaxis(ax, x_categories, graders, items):
    """Simple x-axis customization with just main category labels and dashed separators"""
    
    n_graders = len(graders)
    n_items = len(items)
    
    # Remove default tick labels
    ax.set_xticklabels([])
    
    # Add main LLM category labels
    new_tick_positions = []
    new_tick_labels = []
    
    for i, x_cat in enumerate(x_categories):
        # Center position for this LLM category
        center_pos = i * (n_graders * n_items) + (n_graders * n_items - 1) / 2
        new_tick_positions.append(center_pos)
        new_tick_labels.append(x_cat)
    
    ax.set_xticks(new_tick_positions)
    ax.set_xticklabels(new_tick_labels)
    
    # Add dashed separators between LLM categories only
    for i in range(1, len(x_categories)):
        separator_pos = i * (n_graders * n_items) - 0.5
        ax.axvline(x=separator_pos, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)


def draw_violin_plot_single(ax, df, x_var, color_scheme, x_label='Evaluated model',
                        dodge=False, y_lim=[-0.5, 11.5], legend_title=None,
                        show_predictions=False, idata=None, pred_alpha=0.7,
                        category_maps=None, show_items=False, jitter_strength=0.15):

    box_palette, strip_palette = get_palettes(color_scheme)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    formatted_title = f'Scores by grader (N={len(df)})'
    
    sns.violinplot(
        x=x_var, y='score', hue='grader', data=df,
        palette=box_palette, ax=ax,
        width=PLOT_STYLE["box_width"],
        dodge=dodge,
        inner=None,      
        linewidth=0.1      
    )

    sns.stripplot(
        x=x_var, y='score', hue='grader', data=df,
        palette=strip_palette, ax=ax,
        alpha=PLOT_STYLE["alpha"],
        dodge=dodge,
        jitter=PLOT_STYLE["jitter"],
        size=PLOT_STYLE["point_size"],
        legend=False
    )

    legend = ax.get_legend()
    if legend:
        legend.remove()

    # Optional posterior predictive overlays
    if show_predictions and idata is not None and category_maps is not None:
        add_predictions_to_plot(ax, df, idata, x_var, dodge, pred_alpha, category_maps)

    # Final formatting
    ax.set_title(formatted_title, fontsize=PLOT_STYLE["title_size"])
    ax.set_ylabel('Score', fontsize=PLOT_STYLE["label_size"])
    ax.set_xlabel(x_label, fontsize=PLOT_STYLE["label_size"])
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', labelsize=PLOT_STYLE["tick_size"])


def plot_combined_figure_single(df, idata, var_names, labeller, color_scheme,
                         x_var='grader', x_label='Annotated by',
                         dodge=False, fig_path=None, legend_title=None,
                         y_lim=[-0.5, 11.5], xlim=[-2.5, 2.5], vertical_centering=False,
                         show_predictions=False, pred_alpha=0.7, category_maps=None,
                         show_items=False, jitter_strength=0.15):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=PLOT_STYLE["dpi"])

    draw_violin_plot_single(
        ax=ax1, df=df, x_var=x_var, color_scheme=color_scheme,
        x_label=x_label, dodge=dodge, y_lim=y_lim, legend_title=legend_title,
        show_predictions=show_predictions, idata=idata, pred_alpha=pred_alpha,
        category_maps=category_maps, show_items=show_items, jitter_strength=jitter_strength,
    )

    draw_forest_plot(
        ax=ax2, idata=idata, var_names=var_names, labeller=labeller, 
        xlim=xlim, vertical_centering=vertical_centering
    )

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()

def plot_combined_figure(df, idata, var_names, labeller, color_scheme,
                         x_var='grader', x_label='Annotated by',
                         dodge=False, fig_path=None, legend_title=None,
                         y_lim=[-0.5, 11.5], xlim=[-2.5, 2.5], vertical_centering=False,
                         show_predictions=False, pred_alpha=0.7, category_maps=None,
                         show_items=False, jitter_strength=0.15, legend_fontsize=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=PLOT_STYLE["dpi"])

    draw_violin_plot(
        ax=ax1, df=df, x_var=x_var, color_scheme=color_scheme,
        x_label=x_label, dodge=dodge, y_lim=y_lim, legend_title=legend_title,
        show_predictions=show_predictions, idata=idata, pred_alpha=pred_alpha,
        category_maps=category_maps, show_items=show_items, jitter_strength=jitter_strength,
        legend_fontsize=legend_fontsize
    )

    draw_forest_plot(
        ax=ax2, idata=idata, var_names=var_names, labeller=labeller, 
        xlim=xlim, vertical_centering=vertical_centering
    )

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()


def plot_krippendorff_comparison(alpha_observed, valid_predictions, valid_corrected, 
                                  predictions_mean, predictions_ci, corrected_mean, corrected_ci, 
                                  fig_path=None, trad_s=300, trad_linewidth=2, glm_s=100, glm_linewidth=2):
    """
    Plot comparison of Krippendorff's α approaches: raw observed, GLM predictions, and bias-corrected.

    Args:
        alpha_observed: Raw observed Krippendorff's α.
        valid_predictions: Array of posterior samples for GLM predictions.
        valid_corrected: Array of posterior samples for bias-corrected α.
        predictions_mean: Mean of GLM predictions α.
        predictions_ci: 95% CI for GLM predictions α.
        corrected_mean: Mean of bias-corrected α.
        corrected_ci: 95% CI for bias-corrected α.
        fig_path: Optional path to save the figure.
    """

    plt.figure(figsize=(10, 4), dpi=PLOT_STYLE["dpi"])

    # Positions for the plots
    positions = [2, 1, 0]

    # 1. Raw observed as a single point (top)
    plt.scatter(alpha_observed, positions[0], marker='X', s=trad_s, color='red', 
                edgecolors='darkred', linewidth=trad_linewidth, zorder=10)

    # 2. GLM Predictions distribution (middle)
    parts1 = plt.violinplot([valid_predictions], positions=[positions[1]], vert=False, 
                            widths=0.4, showmeans=False, showmedians=False, showextrema=False)
    parts1['bodies'][0].set_facecolor('lightblue')
    parts1['bodies'][0].set_alpha(0.7)
    parts1['bodies'][0].set_edgecolor('navy')

    # Posterior mean and credible intervals for GLM predictions
    plt.scatter(predictions_mean, positions[1], marker='o', s=glm_s, color='blue', 
                zorder=9, edgecolors='darkblue', linewidth=glm_linewidth)
    plt.plot([predictions_ci[0], predictions_ci[1]], [positions[1], positions[1]], 
             'b-', linewidth=6, alpha=0.4, solid_capstyle='round')

    # 3. Bias-corrected distribution (bottom)
    parts2 = plt.violinplot([valid_corrected], positions=[positions[2]], vert=False, 
                            widths=0.4, showmeans=False, showmedians=False, showextrema=False)
    parts2['bodies'][0].set_facecolor('lightgreen')
    parts2['bodies'][0].set_alpha(0.7)
    parts2['bodies'][0].set_edgecolor('darkgreen')

    # Posterior mean and credible intervals for bias-corrected
    plt.scatter(corrected_mean, positions[2], marker='o', s=glm_s, color='green', 
                zorder=9, edgecolors='darkgreen', linewidth=glm_linewidth)
    plt.plot([corrected_ci[0], corrected_ci[1]], [positions[2], positions[2]], 
             'g-', linewidth=6, alpha=0.4, solid_capstyle='round')

    # Vertical reference lines
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5, zorder=1, label='α = 0 (random agreement)')
    plt.axvline(0.4, color='orange', linestyle=':', alpha=0.7, zorder=1, label='α = 0.4 (moderate agreement)')

    # Labels and formatting
    plt.xlabel("Krippendorff's α", fontsize=PLOT_STYLE["label_size"])
    plt.ylabel('', fontsize=PLOT_STYLE["label_size"])

    # Custom y-tick labels
    y_labels = [
        f'Traditional (point estimate)\n(α = {alpha_observed:.3f})',
        f'Based on GLM predictions\n(mean = {predictions_mean:.3f})',
        f'Based on bias-corrected GLM predictions\n(mean = {corrected_mean:.3f})'
    ]
    plt.yticks(positions, y_labels, fontsize=PLOT_STYLE["tick_size"])

    # Title formatting to match `plot_combined_figure`
    plt.title('Comparison of Krippendorff\'s α Approaches', fontsize=PLOT_STYLE["title_size"])

    # Grid and legend
    plt.grid(True, alpha=0.3, axis='x')
    plt.legend(loc='upper right', fontsize=PLOT_STYLE["legend_size"])

    # Set x-limits with padding
    all_values = list(valid_predictions) + list(valid_corrected) + [alpha_observed]
    x_min, x_max = min(all_values), max(all_values)
    x_range = x_max - x_min
    plt.xlim(x_min - 0.15*x_range, x_max + 0.15*x_range)
    plt.ylim(-0.5, 2.5)

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=PLOT_STYLE["dpi"], bbox_inches='tight')
    plt.show()


def draw_pairwise_grouped_bar_plot(ax, df, color_scheme, x_label='LLM Pair', 
                                  y_lim=[-0.05, 1.1], legend_title='Grader',
                                  show_individual_points=True, show_ci=True):
    
    box_palette, strip_palette = get_palettes(color_scheme)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    
    # Calculate preference rates by pair and grader
    pair_grader_summary = df.groupby(['llm_pair', 'grader']).agg({
        'first_is_chosen': ['mean', 'count']
    }).round(3)
    pair_grader_summary.columns = ['preference_rate', 'n_comparisons']
    pair_grader_summary = pair_grader_summary.reset_index()
    
    # Calculate CI if needed
    if show_ci:
        pair_grader_summary['se'] = np.sqrt(
            pair_grader_summary['preference_rate'] * (1 - pair_grader_summary['preference_rate']) / 
            pair_grader_summary['n_comparisons']
        )
        pair_grader_summary['ci_error'] = 1.96 * pair_grader_summary['se']
    
    # Get unique pairs and graders
    pairs = df['llm_pair'].unique()
    graders = sorted(df['grader'].unique(), key=lambda x: (0 if x.startswith('Human') else 1, x))
    
    # Set up grouped bar positions
    x_pos = np.arange(len(pairs))
    width = 0.35
    
    # Create bars for each grader
    for i, grader in enumerate(graders):
        grader_data = pair_grader_summary[pair_grader_summary['grader'] == grader]
        
        # Get color for this grader
        if isinstance(box_palette, dict):
            color = box_palette[grader]
        else:
            color = box_palette[i % len(box_palette)]
        
        # Create bars
        bars = ax.bar(x_pos + i * width, grader_data['preference_rate'], 
                     width, 
                     yerr=grader_data['ci_error'] if show_ci else None,
                     color=color, alpha=0.7, edgecolor='black', linewidth=0.5,
                     capsize=5, error_kw={'linewidth': 2, 'alpha': 0.8},
                     label=grader)
        
    
    # Add individual data points if requested
    if show_individual_points:
        for i, pair in enumerate(pairs):
            pair_data = df[df['llm_pair'] == pair]
            for j, grader in enumerate(graders):
                grader_data = pair_data[pair_data['grader'] == grader]
                if len(grader_data) > 0:
                    x_jitter = np.random.normal(i + j * width, 0.05, len(grader_data))
                    ax.scatter(x_jitter, grader_data['first_is_chosen'], 
                              color='darkgray', alpha=0.4, s=12, zorder=3)
    
    # Add reference line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Formatting
    title = f'Pairwise preferences by grader (N={len(df)})'
    if show_ci:
        title += ' ± 95% CI'
    ax.set_title(title, fontsize=PLOT_STYLE["title_size"])
    ax.set_ylabel('Preference rate for first LLM', fontsize=PLOT_STYLE["label_size"])
    ax.set_xlabel(x_label, fontsize=PLOT_STYLE["label_size"])
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', labelsize=PLOT_STYLE["tick_size"])
    
    # Set x-axis labels
    ax.set_xticks(x_pos + width/2)
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    
    # Add legend
    ax.legend(title=legend_title, fontsize=PLOT_STYLE["legend_size"]-2,
              title_fontsize=PLOT_STYLE["legend_size"]-2, frameon=False,
              loc='upper right', bbox_to_anchor=(1.03, 0.9))
    


def plot_combined_figure_binomial(df, idata, var_names, labeller, color_scheme,
                         x_var='grader', x_label='LLM pair',
                         dodge=False, fig_path=None, legend_title='Grader',
                         y_lim=[-0.05, 1.1], xlim=[-2.5, 2.5], vertical_centering=False,
                         show_predictions=False, pred_alpha=0.7, category_maps=None,
                         show_items=False, jitter_strength=0.15):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=PLOT_STYLE["dpi"])

    draw_pairwise_grouped_bar_plot(ax=ax1, df=df, color_scheme=color_scheme, 
                                   x_label=x_label, y_lim=y_lim, legend_title=legend_title,
                                  show_individual_points=True, show_ci=True)

    draw_forest_plot(
        ax=ax2, idata=idata, var_names=var_names, labeller=labeller, 
        xlim=xlim, vertical_centering=vertical_centering
    )

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight')
    plt.show()