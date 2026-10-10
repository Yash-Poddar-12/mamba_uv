"""
Read-only sanity check of the merged dataset's COCO annotations vs. the
class folders. Changes nothing on disk. CPU only, takes ~1 min.

Usage (from the mamba-uv root, venv active):
    python tools/inspect_coco.py ^
        --json "C:/Users/Ant pc/Downloads/final_dataset/annotations/instances_default.json" ^
        --images "C:/Users/Ant pc/Downloads/final_dataset/images/default/.data_merged"
"""
import argparse
import collections
import json
import os

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, int(p / 100 * len(sorted_vals)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", required=True, help=".../images/default/.data_merged")
    args = ap.parse_args()

    # ---------------- load ----------------
    try:
        with open(args.json, "r", encoding="utf-8") as f:
            d = json.load(f)
    except json.JSONDecodeError as e:
        with open(args.json, "rb") as f:
            head = f.read(80)
        print(f"!! json.load failed ({e}). First bytes of the file: {head!r}")
        print("   If the file starts with stray characters before '{', that is the problem.")
        return

    images, anns, cats = d["images"], d["annotations"], d["categories"]
    cat_name = {c["id"]: c["name"] for c in cats}
    img_by_id = {im["id"]: im for im in images}
    print(f"images: {len(images)} | annotations: {len(anns)} | categories: {len(cats)}")
    print("\n-- first 2 image entries --")
    for im in images[:2]:
        print(json.dumps(im))
    print("-- first 2 annotation entries --")
    for a in anns[:2]:
        print(json.dumps({k: (v if k != "segmentation" else "...") for k, v in a.items()}))

    # ---------------- folders on disk ----------------
    folders = sorted(e.name for e in os.scandir(args.images) if e.is_dir())
    disk_files = {}   # "class/filename" -> full path
    for cls in folders:
        for fn in os.listdir(os.path.join(args.images, cls)):
            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                disk_files[f"{cls}/{fn}"] = os.path.join(args.images, cls, fn)
    print(f"\nfolders on disk: {len(folders)} | image files on disk: {len(disk_files)}")

    json_names, folder_names = set(cat_name.values()), set(folders)
    print("\n-- category names vs folder names --")
    print("in JSON but no folder :", sorted(json_names - folder_names) or "none")
    print("folder but not in JSON:", sorted(folder_names - json_names) or "none")

    # ---------------- resolve file_name -> file on disk ----------------
    default_dir = os.path.dirname(os.path.normpath(args.images))     # .../images/default
    root_name = os.path.basename(os.path.normpath(args.images))      # .data_merged
    resolved, how = {}, collections.Counter()
    for im in images:
        fn = im["file_name"].replace("\\", "/")
        cands = [fn, fn.split(root_name + "/", 1)[-1]]               # with / without ".data_merged/" prefix
        hit = next((c for c in cands if c in disk_files), None)
        if hit:
            resolved[im["id"]] = hit
            how["class_folder/filename"] += 1
        else:
            how["UNRESOLVED"] += 1
    print(f"\n-- file_name resolution -- {dict(how)}")
    if how["UNRESOLVED"]:
        bad = [im["file_name"] for im in images if im["id"] not in resolved][:5]
        print("examples of unresolved file_name:", bad)
    print("disk images that no JSON image points to:",
          len(set(disk_files) - set(resolved.values())))

    # ---------------- per-image annotation stats ----------------
    per_img = collections.defaultdict(list)
    for a in anns:
        per_img[a["image_id"]].append(a)
    n_ann = collections.Counter(min(len(per_img.get(im["id"], [])), 5) for im in images)
    print("\n-- annotations per image (5 means 5+) --", dict(sorted(n_ann.items())))

    multi_cat = [i for i, al in per_img.items() if len({a["category_id"] for a in al}) > 1]
    print(f"images whose boxes carry >1 distinct category: {len(multi_cat)}")

    # folder label vs annotation category
    agree, disagree = 0, collections.Counter()
    for iid, al in per_img.items():
        if iid not in resolved:
            continue
        folder = resolved[iid].split("/")[0]
        for c in {a["category_id"] for a in al}:
            if cat_name.get(c) == folder:
                agree += 1
            else:
                disagree[(folder, cat_name.get(c))] += 1
    print(f"\n-- folder name vs annotation category -- agree: {agree} | disagree: {sum(disagree.values())}")
    for (f_, c_), n in disagree.most_common(15):
        print(f"   folder={f_!r:35} json_category={c_!r:35} x{n}")

    # box size relative to image
    fracs = []
    for a in anns:
        im = img_by_id.get(a["image_id"])
        if im and im.get("width") and im.get("height"):
            _, _, w, h = a["bbox"]
            fracs.append(w * h / (im["width"] * im["height"]))
    fracs.sort()
    if fracs:
        print("\n-- box area as fraction of image area --")
        print("  p10/p50/p90 = %.2f / %.2f / %.2f" % (pct(fracs, 10), pct(fracs, 50), pct(fracs, 90)))
        print("  boxes covering >90%% of image: %.1f%% | boxes <5%% of image: %.1f%%" % (
            100 * sum(f > 0.9 for f in fracs) / len(fracs), 100 * sum(f < 0.05 for f in fracs) / len(fracs)))


if __name__ == "__main__":
    main()