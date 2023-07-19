import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("-i", "--input", type=str, required=True)
    _ = parser.add_argument("-o", "--output", type=str, required=True)

    args = parser.parse_args()

    vertices: list[tuple[str, str, str]] = []
    normal: list[tuple[str, str, str]] = []
    uv: list[tuple[str, str]] = []
    with open(args.input, "r") as file:
        for line in file:
            strs = [v for v in line.replace(" ", "").strip().strip(",").split(",")]
            floats = [float(v) for v in strs]
            assert len(floats) == 9, f"{len(floats)}"

            vertices.append(tuple(strs[:3]))
            # skip floats[3] which is unused w component.
            normal.append(tuple(strs[4:7]))
            uv.append(tuple(strs[7:9]))

    with open(args.output, "w") as file:
        _ = file.write("vertices = (\n")
        for v in vertices:
            _ = file.write(f"    {v},\n".replace("'", "".replace('"', "")))
        _ = file.write(")\n")

        _ = file.write("normal = (\n")
        for n in normal:
            _ = file.write(f"    {n},\n".replace("'", "".replace('"', "")))
        _ = file.write(")\n")

        _ = file.write("uv = (\n")
        for u in uv:
            _ = file.write(f"    {u},\n".replace("'", "".replace('"', "")))
        _ = file.write(")\n")
