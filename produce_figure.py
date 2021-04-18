#!/usr/bin/env python
import argparse
import itertools
import math
import subprocess
import tqdm
from pathlib import Path

import numpy as np


def make_numerical_tex_string(step, input_size, output_size, padding, kernel_size, stride, dilation, mode):
    """Creates a LaTeX string for a numerical convolution animation.

    Parameters
    ----------
    step : int
        Which step of the animation to generate the LaTeX string for.
    input_size : int
        Convolution input size.
    output_size : int
        Convolution output size.
    padding : int
        Zero padding.
    kernel_size : int
        Convolution kernel size.
    stride : int
        Convolution stride.
    mode : str
        Kernel mode, one of {'convolution', 'average', 'max'}.

    Returns
    -------
    tex_string : str
        A string to be compiled by LaTeX to produce one step of the
        animation.

    """
    if mode not in ('convolution', 'average', 'max'):
        raise ValueError("wrong convolution mode, choices are 'convolution', 'average' or 'max'")
    if dilation != 1:
        raise ValueError("Only a dilation of 1 is currently supported for numerical output")
    max_steps = output_size ** 2
    if step >= max_steps:
        raise ValueError(f'step {step} out of bounds (there are {max_steps} steps for this animation')

    tex_template = Path('latex_templates/numerical_figure.txt').read_text()

    total_input_size = input_size + 2 * padding

    # Always reset the seed
    np.random.seed(1234)
    input_ = np.zeros((total_input_size, total_input_size), dtype='int32')
    input_[padding: padding + input_size, padding: padding + input_size] = \
        np.random.randint(low=0, high=4, size=(input_size, input_size))
    kernel = np.random.randint(low=0, high=3, size=(kernel_size, kernel_size))
    output = np.empty((output_size, output_size), dtype='float32')
    for offset_x, offset_y in itertools.product(range(output_size), range(output_size)):
        if mode == 'convolution':
            output[offset_x, offset_y] = (
                input_[stride * offset_x: stride * offset_x + kernel_size,
                       stride * offset_y: stride * offset_y + kernel_size] * kernel).sum()
        elif mode == 'average':
            output[offset_x, offset_y] = (
                input_[stride * offset_x: stride * offset_x + kernel_size,
                       stride * offset_y: stride * offset_y + kernel_size]).mean()
        else:
            output[offset_x, offset_y] = (
                input_[stride * offset_x: stride * offset_x + kernel_size,
                       stride * offset_y: stride * offset_y + kernel_size]).max()

    offsets = list(itertools.product(range(output_size - 1, -1, -1), range(output_size)))
    offset_y, offset_x = offsets[step]

    if mode == 'convolution':
        kernel_values_string = ''.join(
            "\\node (node) at ({0},{1}) {{\\tiny {2}}};\n".format(
                i + 0.8 + stride * offset_x, j + 0.2 + stride * offset_y, kernel[kernel_size - 1 - j, i])
            for i, j in itertools.product(range(kernel_size), range(kernel_size)))
    else:
        kernel_values_string = '\n'

    return tex_template.format(**{
        'PADDING_TO': f'{total_input_size},{total_input_size}',
        'INPUT_FROM': f'{padding},{padding}',
        'INPUT_TO': f'{padding + input_size},{padding + input_size}',
        'INPUT_VALUES': ''.join(
            "\\node (node) at ({0},{1}) {{\\footnotesize {2}}};\n".format(
                i + 0.5, j + 0.5, input_[total_input_size - 1 - j, i])
            for i, j in itertools.product(range(total_input_size), range(total_input_size))),
        'INPUT_GRID_FROM': f'{stride * offset_x},{stride * offset_y}',
        'INPUT_GRID_TO': f'{stride * offset_x + kernel_size},{stride * offset_y + kernel_size}',
        'KERNEL_VALUES': kernel_values_string,
        'OUTPUT_TO': f'{output_size},{output_size}',
        'OUTPUT_GRID_FROM': f'{offset_x},{offset_y}',
        'OUTPUT_GRID_TO': f'{offset_x + 1},{offset_y + 1}',
        'OUTPUT_VALUES': ''.join(
            "\\node (node) at ({0},{1}) {{\\tiny {2:.1f}}};\n".format(
                i + 0.5, j + 0.5, output[output_size - 1 - j, i])
            for i, j in itertools.product(range(output_size), range(output_size))),
        'XSHIFT': f'{total_input_size + 1}cm',
        'YSHIFT': f'{(total_input_size - output_size) // 2}cm',
    }).encode("utf-8"), output_size


def make_arithmetic_tex_string(step, input_size, output_size, padding, kernel_size, stride, dilation, transposed):
    """Creates a LaTeX string for a convolution arithmetic animation.

    Parameters
    ----------
    step : int
        Which step of the animation to generate the LaTeX string for.
    input_size : int
        Convolution input size.
    output_size : int
        Convolution output size.
    padding : int
        Zero padding.
    kernel_size : int
        Convolution kernel size.
    stride : int
        Convolution stride.
    dilation: int
        Input Dilation
    transposed : bool
        If ``True``, generate strings for the transposed convolution
        animation.

    Returns
    -------
    tex_string : str
        A string to be compiled by LaTeX to produce one step of the
        animation.

    """
    if transposed:
        # Used to add bottom-padding to account for odd shapes
        bottom_pad = (input_size + 2 * padding - kernel_size) % stride
        # Dilation has no meaning in transposed conv
        dilation = 1

        output_size = int(math.floor((input_size + 2 * padding - kernel_size) / stride)) + 1
        if output_size <= 0:
            raise ValueError("Output size is less or equal to zero, which is not allowed.")
        input_size, output_size, padding, spacing, stride = (
            output_size, input_size, kernel_size - 1 - padding, stride, 1)
        total_input_size = output_size + kernel_size - 1
        y_adjustment = 0
    else:
        # Not used in convolutions
        bottom_pad = 0
        # Spacing of input units, 1 means units are tightly adjacent
        spacing = 1

        kernel_size = (kernel_size - 1) * dilation + 1
        # For conv, we omit the provided `output_size` and compute the output size ourselves
        output_size = int(math.floor((input_size + 2 * padding - kernel_size) / stride)) + 1
        if output_size <= 0:
            raise ValueError("Output size is less or equal to zero, which is not allowed.")
        total_input_size = input_size + 2 * padding
        y_adjustment = (total_input_size - (kernel_size - stride)) % stride

    max_steps = output_size ** 2
    if step >= max_steps:
        raise ValueError('step {} out of bounds (there are '.format(step) +
                         '{} steps for this animation'.format(max_steps))

    tex_template = Path('latex_templates/arithmetic_figure.txt').read_text()
    unit_template = '\\draw[draw=base03, fill=blue, thick] ({},{}) rectangle ({},{});'

    offsets = list(itertools.product(range(output_size - 1, -1, -1), range(output_size)))
    offset_y, offset_x = offsets[step]

    return tex_template.format(**{
        'PADDING_TO': f'{total_input_size},{total_input_size}',
        'INPUT_UNITS': ''.join(
            unit_template.format(padding + spacing * i,
                                 bottom_pad + padding + spacing * j,
                                 padding + spacing * i + 1,
                                 bottom_pad + padding + spacing * j + 1)
            for i, j in itertools.product(range(input_size), range(input_size))),
        'INPUT_GRID_FROM_X': f'{stride * offset_x}',
        'INPUT_GRID_FROM_Y': f'{y_adjustment + stride * offset_y}',
        'INPUT_GRID_TO_X': f'{stride * offset_x + kernel_size}',
        'INPUT_GRID_TO_Y': f'{y_adjustment + stride * offset_y + kernel_size}',
        'DILATION': f'{dilation}',
        'OUTPUT_BOTTOM_LEFT': f'{offset_x},{offset_y}',
        'OUTPUT_BOTTOM_RIGHT': f'{offset_x + 1},{offset_y}',
        'OUTPUT_TOP_LEFT': f'{offset_x},{offset_y + 1}',
        'OUTPUT_TOP_RIGHT': f'{offset_x + 1},{offset_y + 1}',
        'OUTPUT_TO': f'{output_size},{output_size}',
        'OUTPUT_GRID_FROM': f'{offset_x},{offset_y}',
        'OUTPUT_GRID_TO': f'{offset_x + 1},{offset_y + 1}',
        'OUTPUT_ELEVATION': f'{total_input_size + 1}cm',
    }).encode("utf-8"), output_size


def make_alphabet_tex_string(step, input_size, output_size, padding, kernel_size, stride, dilation, transposed=False, **kwargs):
    ori_input_size = input_size
    ori_output_size = output_size

    if transposed:
        # Used to add bottom-padding to account for odd shapes
        bottom_pad = (input_size + 2 * padding - kernel_size) % stride
        # Dilation has no meaning in transposed conv
        dilation = 1

        output_size = int(math.floor((input_size + 2 * padding - kernel_size) / stride)) + 1
        if output_size <= 0:
            raise ValueError("Output size is less or equal to zero, which is not allowed.")
        ori_output_size = output_size
        input_size, output_size, padding, spacing, stride = (
            output_size, input_size, kernel_size - 1 - padding, stride, 1)
        total_input_size = output_size + kernel_size - 1
        y_adjustment = 0
    else:
        # Not used in convolutions
        bottom_pad = 0
        # Spacing of input units, 1 means units are tightly adjacent
        spacing = 1

        kernel_size = (kernel_size - 1) * dilation + 1
        # For conv, we omit the provided `output_size` and compute the output size ourselves
        output_size = int(math.floor((input_size + 2 * padding - kernel_size) / stride)) + 1
        if output_size <= 0:
            raise ValueError("Output size is less or equal to zero, which is not allowed.")
        ori_output_size = output_size
        total_input_size = input_size + 2 * padding
        y_adjustment = (total_input_size - (kernel_size - stride)) % stride

    input_length = input_size ** 2
    output_length = output_size ** 2
    ori_input_length = ori_input_size ** 2
    ori_output_length = ori_output_size ** 2
    if step >= output_length:
        raise ValueError(
            'step {} out of bounds (there are {} steps for this animation'.format(step, output_length))

    # Absolute location
    spacing_between_input_and_output = 1
    left_width = max(total_input_size, output_size)
    left_height = total_input_size + spacing_between_input_and_output + output_size
    mid_width = ori_input_length
    mid_height = ori_output_length
    right_width = 2 + 4
    right_height = max(min(ori_input_length, 9), min(ori_output_length, 9))     # Make the right_height depends on the total_height
    total_height = max(left_height, mid_height, right_height)
    left_xy = (0, (total_height - left_height) / 2)
    left_output_xy = ((total_input_size - output_size) / 2, total_input_size + spacing_between_input_and_output)
    mid_xy = (left_width + 2, (total_height - mid_height) / 2)
    right_xy = (left_width + 2 + mid_width, (total_height - right_height) / 2)
    right_input_xy = (right_xy[0] + 2, max(0 if total_height % 2 == 1 else 0.5, (total_height - input_length) / 2))
    right_output_xy = (right_xy[0] + 5, max(0 if total_height % 2 == 1 else 0.5, (total_height - output_length) / 2))

    # Relative location refering to *_xy of absolute locations
    left_total_input_xy0 = (0, 0)
    left_total_input_xy1 = (left_total_input_xy0[0] + total_input_size, left_total_input_xy0[1] + total_input_size)
    left_input_xy0 = (padding, padding)
    left_input_xy1 = (left_input_xy0[0] + input_size, left_input_xy0[1] + input_size)
    left_output_xy0 = (0, 0)
    left_output_xy1 = (left_output_xy0[0] + output_size, left_output_xy0[1] + output_size)
    mid_sparse_xy0 = (0, 0)
    mid_sparse_xy1 = (mid_sparse_xy0[0] + ori_input_length, mid_sparse_xy0[1] + ori_output_length)
    mid_times_xy = (mid_width + 1, mid_height / 2)
    mid_equal_xy = (mid_width + 4, mid_height / 2)
    right_input_xy0 = (0, 0)
    right_input_xy1 = (1, min(total_height if total_height % 2 == 1 else total_height - 1, right_input_xy0[1] + input_length))
    right_output_xy0 = (0, 0)
    right_output_xy1 = (1, min(total_height if total_height % 2 == 1 else total_height - 1, right_output_xy0[1] + output_length))

    if transposed:  # Kernel subindex must be transposed
        ab_input = "b"
        ab_output = "c"
        kernel_subindex = list(itertools.product(range(kernel_size - 1, -1, -1), range(kernel_size)))
    else:
        ab_input = "a"
        ab_output = "b"
        kernel_subindex = list(itertools.product(range(kernel_size), range(kernel_size - 1, -1, -1)))
    ab_kernel = "w"
    input_index = list(itertools.product(range(input_size), range(input_size)))
    kernel_index = list(itertools.product(range(kernel_size), range(kernel_size)))
    output_index = list(itertools.product(range(output_size), range(output_size)))
    input_subindex = list(itertools.product(range(input_size), range(input_size - 1, -1, -1)))
    output_subindex = list(itertools.product(range(output_size), range(output_size - 1, -1, -1)))

    kernel_offsets = list(itertools.product(range(output_size - 1, -1, -1), range(output_size)))
    kernel_offset_y, kernel_offset_x = kernel_offsets[step]

    def get_sparse_index():
        kfv = []
        total_input_size_with_dilation = spacing * (input_size - 1) + 1 + 2 * padding
        ifv = np.zeros((total_input_size_with_dilation,) * 2, dtype=np.uint8)    # input flat values
        has_value_index = list(itertools.product(range(padding, total_input_size_with_dilation - padding, spacing), 
                                                range(padding, total_input_size_with_dilation - padding, spacing)))
        for i, (x, y) in enumerate(has_value_index):
            ifv[x, y] = i + 1
        
        if transposed:
            for i in range(output_size):
                for j in range(output_size):
                    patch = ifv[i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
                    indices = [ori_output_length - 1 - x for x in patch[patch != 0] - 1]
                    subindex = np.where(patch != 0)
                    kfv.append((([i * output_size + j] * len(indices), indices), 
                               (kernel_size - 1 - subindex[0], kernel_size - 1 - subindex[1])))    # ((i, j), kernel_subindex)
        else:
            for i in range(output_size):
                for j in range(output_size):
                    patch = ifv[i * stride:i * stride + kernel_size, j * stride:j * stride + kernel_size]
                    indices = (patch[patch != 0] - 1).tolist()
                    kfv.append(((indices, [ori_output_length - 1 - i * output_size - j] * len(indices)), 
                               np.where(patch != 0)))    # ((i, j), kernel_subindex)
        return kfv

    kernel_flat_values = get_sparse_index()

    def gen_right_string(ab, max_height, templates, index, color):
        rectangle_template, grid_template, value_template, dot_template, step_template = templates
        result = ''
        length = len(index)
        if length > max_height:
            half = (max_height - 1) // 2
            # Lower half
            result += rectangle_template.format(color, 0, 0, 1, half)
            result += ''.join(
                grid_template.format(0, i, 1, i + 1) for i in range(half))
            result += ''.join(
                value_template.format(0.5, i + 0.5, "large", ab, x, y) for i, (x, y) in zip(range(half), reversed(index)))
            # VDots
            result += dot_template.format(0.5, half + 0.6)
            # Upper half
            result += rectangle_template.format(color, 0, half + 1, 1, max_height)
            result += ''.join(
                grid_template.format(0, max_height - 1 - i, 1, max_height - 1 - i + 1) for i in range(half))
            result += ''.join(
                value_template.format(0.5, max_height - 1 - i + 0.5, "large", ab, x, y) for i, (x, y) in zip(range(half), index))
            
            if color == 'cyan':     # for output
                if step < half:
                    result += step_template.format(0, max_height - 1 - step, 1, max_height - 1 - step + 1)
                elif step < length - half:
                    pass
                else:
                    result += step_template.format(0, length - 1 - step, 1, length - 1 - step + 1)
        else:
            max_height = length
            result += rectangle_template.format(color, 0, 0, 1, max_height)
            result += ''.join(
                grid_template.format(0, max_height - 1 - i, 1, max_height - 1 - i + 1) for i in range(max_height))
            result += ''.join(
                value_template.format(0.5, max_height - 1 - i + 0.5, "large", ab, x, y)
                for i, (x, y) in zip(range(max_height), index))
            if color == 'cyan':
                result += step_template.format(0, max_height - 1 - step, 1, max_height - 1 - step + 1)
        return result

    tex_template = Path('latex_templates/alphabet.txt').read_text()
    unit_template = '\\draw[draw=base03, fill=blue, thick] ({},{}) rectangle ({},{});'
    value_template = '\\node (node) at ({},{}) {{\\{} $ {}_{{{}{}}} $}};'
    highlight_template = '\\draw[fill=base02, opacity=0.4] ({},{}) rectangle ({},{});'
    right_templates = [
        '\\draw[fill={}] ({},{}) rectangle ({},{});',
        '\\draw[step=10mm, base03, thick] ({},{}) grid ({},{});',
        value_template,
        '\\node (node) at ({},{}) {{\\large $ \\vdots $}};',
        '\draw[fill=base02, opacity=0.4] ({},{}) rectangle ({},{});'
    ]

    return tex_template.format(**{
        'TOTAL_HEIGHT': f'{total_height}',
        'LEFT_X': f'{left_xy[0]}',
        'LEFT_Y': f'{left_xy[1]}',
        'LEFT_OUTPUT_X': f'{left_output_xy[0]}',
        'LEFT_OUTPUT_Y': f'{left_output_xy[1]}',
        'MID_X': f'{mid_xy[0]}',
        'MID_Y': f'{mid_xy[1]}',
        'RIGHT_X': f'{right_xy[0]}',
        'RIGHT_Y': f'{right_xy[1]}',
        'RIGHT_INPUT_X': f'{right_input_xy[0]}',
        'RIGHT_INPUT_Y': f'{right_input_xy[1]}',
        'RIGHT_OUTPUT_X': f'{right_output_xy[0]}',
        'RIGHT_OUTPUT_Y': f'{right_output_xy[1]}',

        'LEFT_TOTAL_INPUT_XY0': f'{left_total_input_xy0[0]}, {left_total_input_xy0[1]}',
        'LEFT_TOTAL_INPUT_XY1': f'{left_total_input_xy1[0]}, {left_total_input_xy1[1]}',
        'LEFT_INPUT_UNITS': ''.join(
            unit_template.format(left_input_xy0[0] + spacing * i,
                                 bottom_pad + left_input_xy0[0] + spacing * j,
                                 left_input_xy0[0] + spacing * i + 1,
                                 bottom_pad + left_input_xy0[0] + spacing * j + 1)
            for i, j in input_index),
        'LEFT_INPUT_VALUES': ''.join(
            value_template.format(left_input_xy0[0] + spacing * i + 0.4, 
                                  bottom_pad + left_input_xy0[0] + spacing * j + 0.6, 
                                  "large", ab_input, x, y)
            for (i, j), (y, x) in zip(input_index, input_subindex)),
        'LEFT_KERNEL_FROM': f'{kernel_offset_x * stride},{kernel_offset_y * stride + y_adjustment}',
        'LEFT_KERNEL_TO': f'{kernel_offset_x * stride + kernel_size},{kernel_offset_y * stride + kernel_size + y_adjustment}',
        'LEFT_KERNEL_VALUES': ''.join(
            value_template.format(kernel_offset_x * stride + i + 0.75, 
                                  kernel_offset_y * stride + y_adjustment + j + 0.2, 
                                  "scriptsize", ab_kernel, x, y)
            for (i, j), (y, x) in zip(kernel_index, kernel_subindex)),
        'LEFT_OUTPUT_XY0': f'{left_output_xy0[0]},{left_output_xy0[1]}',
        'LEFT_OUTPUT_XY1': f'{left_output_xy1[0]},{left_output_xy1[1]}',
        'LEFT_OUTPUT_STEP_XY0': f'{left_output_xy0[0] + kernel_offset_x},{left_output_xy0[1] + kernel_offset_y}',
        'LEFT_OUTPUT_STEP_XY1': f'{left_output_xy0[0] + kernel_offset_x + 1},{left_output_xy0[1] + kernel_offset_y + 1}',
        'LEFT_OUTPUT_VALUES': ''.join(
            value_template.format(left_output_xy0[0] + i + 0.5, 
                                  left_output_xy0[1] + j + 0.5, 
                                  "large", ab_output, x, y)
            for (i, j), (y, x) in zip(output_index, output_subindex)),
        'MID_SPARSE_XY0': f'{mid_sparse_xy0[0]},{mid_sparse_xy0[1]}',
        'MID_SPARSE_XY1': f'{mid_sparse_xy1[0]},{mid_sparse_xy1[1]}',
        'MID_STEP_XY0': f'{step},{0}' if transposed else f'{0},{ori_output_length - 1 - step}',
        'MID_STEP_XY1': f'{step + 1},{ori_output_length}' if transposed else f'{ori_input_length},{ori_output_length - step}',
        'MID_UNITS': ''.join(
            highlight_template.format(i, j, i + 1, j + 1) for i, j in zip(*kernel_flat_values[step][0])),
        'MID_VALUES': ''.join(
            value_template.format(i + 0.5, j + 0.5, "large", ab_kernel, kernel_ix, kernel_iy)
            for line_input_index, line_kernel_index in kernel_flat_values
            for (i, j), (kernel_ix, kernel_iy) in zip(np.array(line_input_index).transpose(), np.array(line_kernel_index).transpose())
        ),
        'MID_TIMES_XY': f'{mid_times_xy[0]},{mid_times_xy[1]}',
        'MID_EQUAL_XY': f'{mid_equal_xy[0]},{mid_equal_xy[1]}',
        'RIGHT_INPUT_STRING': gen_right_string(ab_input,
                                               right_input_xy1[1],
                                               right_templates,
                                               input_index,
                                               'blue'),
        'RIGHT_OUTPUT_STRING': gen_right_string(ab_output, 
                                                right_output_xy1[1],
                                                right_templates,
                                                output_index,
                                                'cyan'),
    }).encode("utf-8"), output_size


def compile_figure(which_, name, step, **kwargs):
    anim = kwargs.pop("animation")
    dtype = kwargs.pop("type")
    quality = kwargs.pop("quality")
    quality_anim = kwargs.pop("quality_animation")
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    def cvt_dtype(pdfname):
        if dtype == "png":
            pngname = pdfname.with_suffix(".png")
            subprocess.call(f'convert -density {quality_anim if anim else quality} {pdfname} {pngname}'.split())
            subprocess.call(f'rm {pdfname}'.split())

    def run(step, **kwargs):
        if which_ == 'arithmetic':
            tex_string, output_size = make_arithmetic_tex_string(step, **kwargs)
        elif which_ == 'numerical':
            tex_string, output_size = make_numerical_tex_string(step, **kwargs)
        elif which_ == 'alphabet':
            tex_string, output_size = make_alphabet_tex_string(step, **kwargs)
        else:
            raise ValueError()
        
        jobname = '{}_{:02d}'.format(name, step)
        p = subprocess.Popen(['pdflatex', f'-jobname={jobname}', '-output-directory', str(out_dir)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdoutdata, _ = p.communicate(input=tex_string)
        # Remove logs and aux if compilation was successfull
        if '! LaTeX Error' in stdoutdata.decode("utf-8") or '! Emergency stop' in stdoutdata.decode("utf-8"):
            print(f'! LaTeX Error: check the log file in {out_dir}/{jobname}.log')
        else:
            subprocess.call(['rm'] + list(out_dir.glob("*.aux")) + list(out_dir.glob("*.log")))

        pdfname = out_dir / (jobname + ".pdf")
        cvt_dtype(pdfname)
        return output_size

    output_size = run(step, **kwargs)

    if anim:
        gif_dir = Path(__file__).parent / "gif"
        gif_dir.mkdir(parents=True, exist_ok=True)
        print("Generating frames...")
        for i in tqdm.tqdm(range(output_size ** 2)):
            if i != step:
                run(i, **kwargs)
        print("Synthesizing gif... (If it takes too much time, try to reduce output quality by add `-qa 150`)")
        input_files = ' '.join([str(x) for x in sorted(out_dir.glob("*.png"))])
        out_file = gif_dir / (name + ".gif")
        subprocess.call(f'convert -delay 100 -loop 0 -layers Optimize +map -background white -alpha remove -alpha off -dispose previous  {input_files} {out_file}'.split())
        subprocess.call(f'gifsicle --batch -O3 {out_file}'.split())
        print("Clear working space...")
        subprocess.call(f'rm {" ".join([str(x) for x in out_dir.glob("*.png")])}'.split())
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile a LaTeX figure as part of a convolution animation.")

    subparsers = parser.add_subparsers()

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("name", type=str, help="name for the animation")
    parent_parser.add_argument("-e", "--step", type=int, default=0, help="animation step. (default: %(default)s)")
    parent_parser.add_argument("-i", "--input-size", type=int, default=5, help="input size. (default: %(default)s)")
    parent_parser.add_argument("-o", "--output-size", type=int, default=3, help="output size. (default: %(default)s)")
    parent_parser.add_argument("-p", "--padding", type=int, default=0, help="zero padding. (default: %(default)s)")
    parent_parser.add_argument("-k", "--kernel-size", type=int, default=3, help="kernel size. (default: %(default)s)")
    parent_parser.add_argument("-s", "--stride", type=int, default=1, help="stride. (default: %(default)s)")
    parent_parser.add_argument("-d", "--dilation", type=int, default=1, help="dilation. (default: %(default)s)")
    parent_parser.add_argument("-a", "--animation", action="store_true",
                               help="Make an animation output instead of a single step pdf.")
    parent_parser.add_argument("-y", "--type", type=str, default="png",  choices=["pdf", "png"], help="Output type of a single frame. (default: %(default)s)")
    parent_parser.add_argument("-q", "--quality", type=int, default=600, help="Quality of the frame. Larger is better. (default: %(default)s)")
    parent_parser.add_argument("-qa", "--quality-animation", type=int, default=300, help="Quality of the animation. Larger is better. (default: %(default)s)")

    subparser = subparsers.add_parser('arithmetic', parents=[parent_parser], help='convolution arithmetic animation')
    subparser.add_argument("-t", "--transposed", action="store_true", help="use a transposed convolution")
    subparser.set_defaults(which_='arithmetic')

    subparser = subparsers.add_parser('numerical', parents=[parent_parser], help='numerical convolution animation')
    subparser.add_argument("-m", "--mode", type=str, default='convolution', choices=('convolution', 'average', 'max'), help="kernel mode (default: %(default)s)")
    subparser.set_defaults(which_='numerical')

    subparser = subparsers.add_parser('alphabet', parents=[parent_parser], help='alphabet convolution animation in a matrix multiplication view')
    subparser.add_argument("-t", "--transposed", action="store_true", help="use a transposed convolution")
    subparser.set_defaults(which_='alphabet')

    args = parser.parse_args()
    args_dict = vars(args)
    if len(args_dict) == 0:
        parser.print_help()
    else:
        which_ = args_dict.pop('which_')
        name = args_dict.pop('name')
        step = args_dict.pop('step')

        compile_figure(which_, name, step, **args_dict)
