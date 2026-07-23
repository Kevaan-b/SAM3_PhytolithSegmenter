import { describe, expect, it } from "vitest";
import {
  adjacentFolder,
  adjacentImage,
  findFolder,
  firstFolderWithImages,
  flattenFolders,
  folderBreadcrumbs,
  siblingFolders,
  type DataFolder,
  type DataImage,
} from "./data-navigator";

function image(name: string, path: string): DataImage {
  return { id: path, name, path, url: `/data/${path}`, cacheState: "ready" };
}

const tree: DataFolder = {
  name: "Data",
  path: "",
  images: [],
  folders: [
    {
      name: "test",
      path: "test",
      images: [],
      folders: [],
    },
    {
      name: "train",
      path: "train",
      images: [
        image("a.png", "train/a.png"),
        image("b.png", "train/b.png"),
      ],
      folders: [
        {
          name: "nested",
          path: "train/nested",
          images: [
            image("c.png", "train/nested/c.png"),
          ],
          folders: [],
        },
      ],
    },
    {
      name: "val",
      path: "val",
      images: [
        image("v.png", "val/v.png"),
      ],
      folders: [],
    },
  ],
};

describe("recursive data navigation", () => {
  it("flattens and resolves folders at any depth", () => {
    expect(flattenFolders(tree).map(({ path }) => path)).toEqual([
      "",
      "test",
      "train",
      "train/nested",
      "val",
    ]);
    expect(findFolder(tree, "train/nested")?.name).toBe("nested");
    expect(firstFolderWithImages(tree)?.path).toBe("train");
  });

  it("moves only between sibling folders at the current level", () => {
    const train = findFolder(tree, "train")!;
    expect(siblingFolders(tree, train).map(({ path }) => path)).toEqual([
      "test",
      "train",
      "val",
    ]);
    expect(adjacentFolder(tree, train, -1)?.path).toBe("test");
    expect(adjacentFolder(tree, train, 1)?.path).toBe("val");
    expect(adjacentFolder(tree, findFolder(tree, "train/nested")!, 1)).toBe(
      undefined,
    );
  });

  it("moves only between direct images in the selected folder", () => {
    const train = findFolder(tree, "train")!;
    expect(adjacentImage(train, train.images[0]!, 1)?.path).toBe(
      "train/b.png",
    );
    expect(adjacentImage(train, train.images[1]!, 1)).toBe(undefined);
  });

  it("builds clickable breadcrumbs for nested folders", () => {
    expect(
      folderBreadcrumbs(tree, findFolder(tree, "train/nested")!).map(
        ({ path }) => path,
      ),
    ).toEqual(["", "train", "train/nested"]);
  });
});
