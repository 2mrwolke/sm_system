import numpy as np
from matplotlib import pyplot as plt


def plot_bode_measurement(
    freq_gt: np.ndarray, 
    mag_gt: np.ndarray, 
    phase_gt: np.ndarray, 
    sampling_rate: float,
    freq_mag: np.ndarray = None, 
    mag: np.ndarray = None, 
    freq_phase: np.ndarray = None, 
    phase: np.ndarray = None, 
    plot_title: str = 'Bode Diagram', 
    plot_labels: list = ['Theoretic', 'Calculated'], 
    legend_location: str = 'best', 
    colors: list = None,
    limits: list = None,
    save: bool = False, 
    save_individual: bool = False,
    path: str = None,
) -> tuple:
    """
    Plots a Bode diagram with magnitude and phase plots.

    Parameters:
    - freq_gt: np.ndarray - Frequencies for the theoretical Bode plot.
    - mag_gt: np.ndarray - Magnitude values for the theoretical Bode plot.
    - phase_gt: np.ndarray - Phase values for the theoretical Bode plot.
    - sampling_rate: float - Sampling rate of the system.
    - freq_mag: np.ndarray, optional - Frequencies for the calculated magnitude plot.
    - mag: np.ndarray, optional - Magnitude values for the calculated plot.
    - freq_phase: np.ndarray, optional - Frequencies for the calculated phase plot.
    - phase: np.ndarray, optional - Phase values for the calculated plot.
    - plot_title: str, optional - Title of the plot.
    - plot_labels: list, optional - Labels for the legend.
    - legend_location: str, optional - Location of the legend.
    - colors: list, optional - Colors for the plots.
    - limits: list, optional - Limits for the y-axis of magnitude and phase plots.
    - save: bool, optional - Whether to save the plot.
    - save_individual: bool, optional - Whether to save individual plots.

    Returns:
    - tuple: A tuple containing the figure and axes objects.
    """
    
    # Set default values if not provided
    colors = colors or ['tab:blue', 'tab:red', 'tab:green', 'orange']
    limits = limits or [[-60, 12], [-180, 180]]
    path = path or ''
    
    # Create the main figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    # Plot magnitude
    ax[0].grid(which='both', axis='both')
    ax[0].minorticks_on()
    ax[0].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax[0].semilogx(freq_gt, mag_gt, color=colors[0], label=plot_labels[0][0])  # Bode magnitude plot
    if freq_mag is not None and mag is not None:
        ax[0].scatter(freq_mag, mag, color=colors[1], label=plot_labels[0][1])  # Scatter plot of magnitude measurements
    ax[0].set_title('Magnitude', fontsize=22)
    ax[0].set_xlabel('Frequency [Hz]', fontsize=20)
    ax[0].set_ylabel('Magnitude [dB]', fontsize=20)
    ax[0].set_xlim([10, sampling_rate / 2 * 1.25])
    ax[0].set_ylim(limits[0])
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    ax[0].axvline(x=sampling_rate / 2, color='grey', linestyle='-')#, label='Nyquist freq')
    ax[0].text(sampling_rate // 2, limits[0][0], f'{sampling_rate / 2000:.0f} kHz    ', rotation=-90, verticalalignment='bottom')
     
    ax[0].legend(fontsize=20, loc=legend_location)
    
    # Plot phase
    ax[1].grid(which='both', axis='both')
    ax[1].minorticks_on()
    ax[1].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    ax[1].semilogx(freq_gt, phase_gt, color=colors[2], label=plot_labels[1][0])  # Bode phase plot
    if freq_phase is not None and phase is not None:
        ax[1].scatter(freq_phase, phase, color=colors[3], label=plot_labels[1][1])  # Scatter plot of phase measurements
    ax[1].set_title('Phase', fontsize=22)
    ax[1].set_xlabel('Frequency [Hz]', fontsize=20)
    ax[1].set_ylabel('Phase [deg]', fontsize=20)
    ax[1].set_xlim([10, sampling_rate / 2 * 1.25])
    ax[1].set_ylim(limits[1])
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    ax[1].axvline(x=sampling_rate / 2, color='grey', linestyle='-')#, label='Nyquist freq')
    ax[1].text(sampling_rate // 2, limits[1][0], f'{sampling_rate / 2000:.0f} kHz    ', rotation=-90, verticalalignment='bottom')
        
    ax[1].legend(fontsize=20, loc=legend_location)
    
    fig.suptitle(plot_title, fontsize=22, y=1.03)
    plt.subplots_adjust(top=0.9, wspace=0.3)
    
    # Save individual plots if required
    if save and save_individual:
        for i, subplot_title in enumerate(['Magnitude - ' + plot_title, 'Phase - ' + plot_title]):
            individual_fig, individual_ax = plt.subplots(figsize=(10, 6))
            if i == 0:
                individual_ax.semilogx(freq_gt, mag_gt, color=colors[0], label=plot_labels[i][0])
                if freq_mag is not None and mag is not None:
                    individual_ax.scatter(freq_mag, mag, color=colors[1], label=plot_labels[i][1])
                individual_ax.set_ylabel('Magnitude [dB]', fontsize=20)
            else:
                individual_ax.semilogx(freq_gt, phase_gt, color=colors[2], label=plot_labels[i][0])
                if freq_phase is not None and phase is not None:
                    individual_ax.scatter(freq_phase, phase, color=colors[3], label=plot_labels[i][1])
                individual_ax.set_ylabel('Phase [deg]', fontsize=20)
            
            individual_ax.set_title(subplot_title, fontsize=22)
            individual_ax.set_xlabel('Frequency [Hz]', fontsize=20)
            
            individual_ax.axvline(x=sampling_rate / 2, color='grey', linestyle='-')#, label='Nyquist freq')
            individual_ax.text(sampling_rate // 2, limits[i][0], f'{sampling_rate / 2000:.0f} kHz    ', rotation=-90, verticalalignment='bottom')
        
            individual_ax.set_xlim([10, sampling_rate / 2 * 1.25])
            individual_ax.set_ylim(limits[i])
            individual_ax.grid(which='both', axis='both', linestyle='--', linewidth=0.5)
            individual_ax.legend(fontsize=20, loc=legend_location)
            individual_ax.minorticks_on()
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            individual_fig.savefig(f"{path}{plot_title}_{subplot_title.split(' ')[0]}.png", dpi=600, bbox_inches='tight')
            plt.close(individual_fig)
    
    # Save the main figure or show it
    if save and not save_individual:
        plt.savefig(path+plot_title, dpi=600, bbox_inches='tight')
    elif not save:
        plt.show()
    
    return fig, ax
