import argparse
import matplotlib
import numpy as np
import os
import subprocess as SP
import tempfile
import time

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation


COORDS = dict(
    y0=np.array([295, 258, 218, 180, 141, 104]),
    y1=np.array([310.7, 266.7, 220, 177, 131.2, 86.8]),
    x=np.array([120, 267, 462, 650, 815, 983, 1130, 1275]))


def create_xy_grid(x, y0, y1):
    m = (y1 - y0) / (x[-1] - x[0])
    y = np.array([mi * (x - x[0]) + yi for mi, yi in zip(m, y0)])
    x = np.array([x for n in range(6)])
    return x, y


def render(fret_idxs, fps, background_file, output_file,
           next_fret_idxs=None, title='', dpi=300, print_freq=100):
    """Render fretboard indices to video.

    Parameters
    ----------
    fret_idxs : np.ndarrray, shape=(N, 6)
        Active frets in each frame; -1 indicates an off-string.
    fps : scalar
        Video framerate, in Hertz.
    output_file : str
        Path to write the resulting video.
    title : str, default=''
        An optional title for the video.
    dpi : int, default=120
        Resolution of the video (dots per inch).
    background_file : str, module default
        Image file to use as a background.
    print_freq : int, default=100
        Print progress with the given frequency.
    """
    if next_fret_idxs is None:
        next_fret_idxs = (np.zeros_like(fret_idxs) - 1).astype(int)

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=title,
                    artist='Matplotlib',
                    comment='Fretboard Test!')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    # Create base figure.
    fig = plt.figure(figsize=(10, 4))
    ax = fig.gca()
    img_data = plt.imread(background_file)
    extent = (0, img_data.shape[1], img_data.shape[0], 0)
    ax.imshow(img_data, extent=extent)
    ax.set_axis_off()
    plt.tight_layout()

    # Create string highlighters
    dot_color = np.array([102., 204., 255.]).reshape(1, 3)/256.0
    dot_params = dict(marker='o', s=175, c=dot_color, alpha=0.75)
    strings = [ax.scatter(-100, -100, **dot_params) for n in range(6)]

    dot_color = np.array([255., 255., 255.]).reshape(1, 3)/256.0
    dot_params = dict(marker='o', s=175, c=dot_color, alpha=0.33)
    next_strings = [ax.scatter(-100, -100, **dot_params) for n in range(6)]

    x, y = create_xy_grid(**COORDS)
    with writer.saving(fig, output_file, dpi):
        # For each frame
        for frame_idx, frets in enumerate(fret_idxs):
            next_frets = next_fret_idxs[frame_idx]
            # For each string-fret tuple
            for string_idx, fret_idx in enumerate(frets):
                next_fret_idx = next_frets[string_idx]
                str_hook = strings[string_idx]
                if fret_idx >= 0:
                    str_hook.set_visible(True)
                    str_hook.set_offsets([x[string_idx, fret_idx],
                                          y[string_idx, fret_idx]])
                else:
                    str_hook.set_visible(False)

                next_str_hook = next_strings[string_idx]
                if next_fret_idx >= 0:
                    next_str_hook.set_visible(True)
                    next_str_hook.set_offsets([x[string_idx, next_fret_idx],
                                               y[string_idx, next_fret_idx]])
                else:
                    next_str_hook.set_visible(False)

            plt.draw()
            writer.grab_frame(pad_inches=0)
            if (frame_idx % print_freq) == 0:
                print("[{0}] Finished {1} / {2} frames."
                      "".format(time.asctime(), frame_idx, len(fret_idxs)))


def sync_avfiles(video_file, audio_file, output_file):
    """Synchronize an audio and video file.

    Parameters
    ----------
    video_file : str
        Path to an input video file.
    audio_file : str
        Path to an input audio file.
    output_file : str
        Path to write the resulting mixdown.
    """
    args = ['ffmpeg', '-i', video_file, '-i', audio_file, '-c', 'copy',
            '-map', '0:0', '-map', '1:0', output_file]

    proc = SP.Popen(args, stdout=SP.PIPE, stderr=SP.PIPE)
    stdout, stderr = proc.communicate('y')
    if stderr:
        SP.CalledProcessError(proc.returncode, cmd=" ".join(args),
                              output=stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("fret_index_file",
                        metavar="fret_index_file", type=str,
                        help="Filepath to a npy file of fret indexes, shaped "
                             "(N, 6); -1 represents an off-string.")
    parser.add_argument("fps",
                        metavar="fps", type=float, default=12,
                        help="Framerate of the given fret indices.")
    parser.add_argument("background_file",
                        metavar="background_file", type=str,
                        help="Filepath to an image to use as a background.")
    parser.add_argument("output_file",
                        metavar="output_file", type=str,
                        help="File path to save output movie.")
    parser.add_argument("--audio_file",
                        metavar="audio_file", type=str, default='',
                        help="")
    parser.add_argument("--next_fret_file",
                        metavar="next_fret_file", type=str, default='',
                        help="")
    parser.add_argument("--dpi",
                        metavar="dpi", type=int, default=300,
                        help="")

    args = parser.parse_args()
    next_frets = np.load(args.next_fret_file) if args.next_fret_file else None
    render(np.load(args.fret_index_file), args.fps, args.background_file,
           args.output_file, next_fret_idxs=next_frets, dpi=args.dpi)

    if args.audio_file:
        fext = os.path.splitext(args.output_file)[-1]
        fid, tmpfile = tempfile.mkstemp(suffix=".{0}".format(fext))
        os.close(fid)
        if os.path.exists(tmpfile):
            os.remove(tmpfile)
        sync_avfiles(args.output_file, args.audio_file, tmpfile)
        os.rename(tmpfile, args.output_file)
