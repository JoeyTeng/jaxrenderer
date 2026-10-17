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

            vertices.append((strs[0], strs[1], strs[2]))
            # skip floats[3] which is unused w component.
            normal.append((strs[4], strs[5], strs[6]))
            uv.append((strs[7], strs[8]))

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
