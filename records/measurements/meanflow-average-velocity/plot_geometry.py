"""Plot exact Gaussian MeanFlow geometry and one-step Euler on a new output directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
if args.output_dir.exists():
    parser.error('output directory already exists; choose a new directory')
args.output_dir.mkdir(parents=True, exist_ok=False)

# Cache stays in an owned, temporary directory and does not need user-home writes.
with tempfile.TemporaryDirectory(prefix='mpl-cache-', dir=args.output_dir) as cache:
    os.environ['MPLCONFIGDIR'] = cache
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    m, s, z = 1.5, 0.7, -1.5
    tau = np.linspace(1.0, 0.0, 401, dtype=np.float64)
    q = (1-tau)**2*s*s + tau*tau
    a = (1-tau)*m
    exact = a + np.sqrt(q)*z
    exact_endpoint = m+s*z
    instant_velocity_at_noise = z-m
    average_velocity = z-exact_endpoint
    chord = z + (tau-1)*average_velocity
    euler = z + (tau-1)*instant_velocity_at_noise
    derivative = -m + ((1+s*s)*tau-s*s)/np.sqrt(q)*z
    ode_velocity = -m + ((1+s*s)*tau-s*s)/q*(exact-a)
    input_z = np.linspace(-2, 2, 401, dtype=np.float64)
    average_outputs = m+s*input_z
    euler_outputs = np.full_like(input_z, m)
    nine_z = np.linspace(-2, 2, 9, dtype=np.float64)
    nine_average = m+s*nine_z
    nine_euler = np.full_like(nine_z, m)

    np.testing.assert_allclose(derivative, ode_velocity, rtol=0, atol=1e-14)
    np.testing.assert_allclose([exact[0], chord[0], euler[0]], z, rtol=0, atol=1e-14)
    np.testing.assert_allclose([exact[-1], chord[-1]], exact_endpoint, rtol=0, atol=1e-14)
    np.testing.assert_allclose(euler[-1], m, rtol=0, atol=1e-14)
    np.testing.assert_allclose(derivative[0], instant_velocity_at_noise, rtol=0, atol=1e-14)
    assert np.all(q >= s*s/(1+s*s)-1e-14)
    assert float(np.max(np.abs(exact-chord))) > 0.3
    assert abs(average_velocity-(-1.95)) < 1e-14
    assert abs(instant_velocity_at_noise-(-3.0)) < 1e-14
    probe = np.linspace(.01, .99, 31)
    def path(t):
        return (1-t)*m + np.sqrt((1-t)**2*s*s+t*t)*z
    eps=1e-6
    finite_difference=(path(probe+eps)-path(probe-eps))/(2*eps)
    analytic=-m+((1+s*s)*probe-s*s)/np.sqrt((1-probe)**2*s*s+probe*probe)*z
    finite_difference_error=float(np.max(np.abs(finite_difference-analytic)))
    assert finite_difference_error < 1e-8

    np.savez(args.output_dir/'data.npz', tau=tau, exact_trajectory=exact,
             average_endpoint_chord=chord, one_euler_line=euler,
             initial_z=input_z, average_endpoint_map=average_outputs,
             euler_endpoint_map=euler_outputs, nine_z=nine_z,
             nine_average=nine_average, nine_euler=nine_euler)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11.5,
        'axes.titlesize': 13, 'axes.labelsize': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#57606a', 'axes.labelcolor': '#24292f',
        'xtick.color': '#57606a', 'ytick.color': '#57606a',
        'grid.color': '#d8dee4', 'grid.linewidth': .7,
        'svg.hashsalt': 'genai-study-meanflow-gaussian-v1',
    })
    blue, green, orange = '#0072B2', '#009E73', '#D55E00'
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4))
    fig.subplots_adjust(left=.07, right=.985, bottom=.17, top=.81, wspace=.29)
    fig.suptitle('Average velocity specifies a finite displacement', fontsize=17,
                 fontweight='bold', y=.965, color='#24292f')
    fig.text(.5,.91, r'Exact Gaussian example: $m=1.5$, $s=0.7$; generation runs from $\tau=1$ to $\rho=0$',
             ha='center', va='center', color='#57606a', fontsize=11.5)
    ax=axes[0]
    ax.plot(tau,exact,color=blue,lw=2.7,label='Exact ODE trajectory',zorder=3)
    ax.plot(tau,chord,color=green,lw=2.3,ls=(0,(5,3)),
            label=r'Full-interval endpoint chord ($\rho=0$)',zorder=2)
    ax.plot(tau,euler,color=orange,lw=2.1,ls=(0,(2,2)),
            label='One Euler step (initial tangent)',zorder=2)
    ax.scatter([1],[z],color='#24292f',s=46,zorder=6)
    ax.scatter([0],[exact_endpoint],s=83,facecolors='white',edgecolors=green,lw=2.0,zorder=6)
    ax.scatter([0],[m],s=54,color=orange,marker='s',zorder=6)
    ax.annotate(r'$x_1=-1.5$',xy=(1,z),xytext=(.91,-1.32),fontsize=11,
                color='#24292f',ha='left')
    ax.annotate(r'$x_0=0.45$',xy=(0,exact_endpoint),xytext=(.35,.36),
                color=green,fontsize=11,
                arrowprops={'arrowstyle':'-','color':green,'lw':.9})
    ax.annotate(r'Euler: $x_0=1.5$',xy=(0,m),xytext=(.01,1.65),
                color=orange,fontsize=11,ha='right',
                arrowprops={'arrowstyle':'-','color':orange,'lw':.9})
    ax.set(xlim=(1.035,-.045),ylim=(-1.68,1.75),xlabel=r'Time $\tau$  (generation: $1 \rightarrow 0$)',
           ylabel=r'State $x_\tau$')
    ax.set_xticks([1,.75,.5,.25,0]); ax.grid(alpha=.7); ax.set_axisbelow(True)
    ax.set_title(r'A   One starting state: $z=-1.5$',loc='left',pad=12,fontweight='bold')
    ax.legend(loc='upper left',frameon=True,fontsize=10.1,
              facecolor='white',edgecolor='#d8dee4',framealpha=.96,
              handlelength=2.8,labelspacing=.55)

    ax=axes[1]
    ax.plot(input_z,average_outputs,color=green,lw=2.7,label=r'Average velocity: $x_0=m+sz$')
    ax.plot(input_z,euler_outputs,color=orange,lw=2.1,ls=(0,(2,2)),label=r'One Euler: $x_0=m$')
    ax.scatter(nine_z,nine_average,s=30,facecolors='white',edgecolors=green,lw=1.5,zorder=4)
    ax.scatter(nine_z,nine_euler,s=23,color=orange,marker='s',zorder=4)
    ax.set(xlim=(-2.12,2.12),ylim=(-.08,3.08),xlabel=r'Noise state $z=x_1$',ylabel=r'Output state $x_0$')
    ax.set_xticks([-2,-1,0,1,2]); ax.grid(alpha=.7); ax.set_axisbelow(True)
    ax.set_title('B   One evaluation, different endpoint maps',loc='left',
                 fontsize=12.1,pad=12,fontweight='bold')
    ax.text(.04,.94,r'For $Z\sim\mathcal{N}(0,1)$:'+'\n'+'population variance:\naverage 0.49; Euler 0',
            transform=ax.transAxes,ha='left',va='top',fontsize=11,
            bbox={'boxstyle':'round,pad=.35','fc':'white','ec':'#d8dee4','alpha':.96})
    ax.legend(loc='lower right',frameon=True,fontsize=10.5,
              facecolor='white',edgecolor='#d8dee4',framealpha=.96,
              handlelength=2.8,labelspacing=.55)
    fig.text(.07,.035,'A: The dashed chord displays only the full-interval displacement.',
             ha='left',fontsize=10,color='#57606a')
    fig.text(.565,.035,'B: Nine fixed markers; variances follow from the Gaussian population.',
             ha='left',fontsize=9.4,color='#57606a')
    fig.savefig(args.output_dir/'meanflow-gaussian.png',dpi=160,facecolor='white',
                metadata={'Software':f'genai_study; Matplotlib {matplotlib.__version__}'})
    fig.savefig(args.output_dir/'meanflow-gaussian.svg',facecolor='white',
                metadata={'Title':'MeanFlow Gaussian displacement and one-step Euler',
                          'Description':'Two panels show the exact ODE trajectory, endpoint chord, and initial-tangent Euler step; then compare endpoint maps m+s*z and m. The dashed chord illustrates net displacement. Population variances assume a standard normal input.',
                          'Creator':'genai_study original analytic illustration', 'Date':None})
    plt.close(fig)

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    evidence={
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Original analytic scientific illustration; no model inference, training or random sampling.',
        'parameters':{'m':m,'s':s,'z_panel_a':z},
        'geometry':{'time_coordinates':'tau data-to-noise, shown descending 1 to0 for generation',
                    'initial_state':z,'exact_endpoint':exact_endpoint,'one_euler_endpoint':m,
                    'average_velocity':average_velocity,'initial_instantaneous_velocity':instant_velocity_at_noise,
                    'exact_trajectory_max_distance_from_chord':float(np.max(np.abs(exact-chord))),
                    'chord_scope':'Intermediate chord points only illustrate the net endpoint displacement.'},
        'independent_checks':{'derivative_matches_ode_max_abs':float(np.max(np.abs(derivative-ode_velocity))),
                              'derivative_central_difference_max_abs':finite_difference_error,
                              'all_endpoints_and_initial_tangent_assertions_passed':True},
        'endpoint_map':{'fixed_plot_inputs':9,'dense_curve_inputs':401,
                        'population_assumption':'Z~N(0,1)',
                        'population_variance_average':s*s,'population_variance_one_euler':0.,
                        'maximum_endpoint_error_on_nine_fixed_inputs':float(np.max(np.abs(nine_euler-nine_average))),
                        'variance_is_not_estimated_from_plot_markers':True},
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        'script_sha256':sha(__file__),
        'files':{name:{'sha256':sha(args.output_dir/name),'bytes':(args.output_dir/name).stat().st_size}
                 for name in ['data.npz','meanflow-gaussian.png','meanflow-gaussian.svg']},
    }
    (args.output_dir/'verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))
