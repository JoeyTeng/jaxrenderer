import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)

    args = parser.parse_args()

    vertices: list[tuple[str, str, str]] = []
    normal: list[tuple[str, str, str]] = []
    uv: list[tuple[str, str]] = []
    with open(args.input, 'r') as file:
        for line in file:
            strs = [
                v for v in line.replace(' ', '').strip().strip(',').split(',')
            ]
            floats = [float(v) for v in strs]
            assert len(floats) == 9, f"{len(floats)}"

            vertices.append(tuple(strs[:3]))
            # skip floats[3] which is unused w component.
            normal.append(tuple(strs[4:7]))
            uv.append(tuple(strs[7:9]))

    with open(args.output, 'w') as file:
        file.write('vertices = (\n')
        for v in vertices:
            file.write(f'    {v},\n'.replace("'", "".replace('"', '')))
        file.write(')\n')

        file.write('normal = (\n')
        for n in normal:
            file.write(f'    {n},\n'.replace("'", "".replace('"', '')))
        file.write(')\n')

        file.write('uv = (\n')
        for u in uv:
            file.write(f'    {u},\n'.replace("'", "".replace('"', '')))
        file.write(')\n')
