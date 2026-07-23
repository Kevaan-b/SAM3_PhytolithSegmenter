export interface DataImage {
  id: string;
  name: string;
  path: string;
  url: string;
  cacheState: "missing" | "queued" | "encoding" | "ready";
}

export interface DataFolder {
  name: string;
  path: string;
  folders: DataFolder[];
  images: DataImage[];
}

export type NavigationMode = "folder" | "image";

export function flattenFolders(root: DataFolder): DataFolder[] {
  return [root, ...root.folders.flatMap(flattenFolders)];
}

export function findFolder(
  root: DataFolder,
  path: string,
): DataFolder | undefined {
  return flattenFolders(root).find((folder) => folder.path === path);
}

export function firstFolderWithImages(
  root: DataFolder,
): DataFolder | undefined {
  return flattenFolders(root).find((folder) => folder.images.length > 0);
}

export function siblingFolders(
  root: DataFolder,
  folder: DataFolder,
): DataFolder[] {
  if (folder.path === root.path) return [root];
  const parent = findFolder(root, parentPath(folder.path));
  return parent?.folders ?? [];
}

export function adjacentFolder(
  root: DataFolder,
  folder: DataFolder,
  offset: -1 | 1,
): DataFolder | undefined {
  return adjacentItem(
    siblingFolders(root, folder),
    folder.path,
    offset,
    (item) => item.path,
  );
}

export function adjacentImage(
  folder: DataFolder,
  image: DataImage | null,
  offset: -1 | 1,
): DataImage | undefined {
  if (!image) return undefined;
  return adjacentItem(
    folder.images,
    image.path,
    offset,
    (item) => item.path,
  );
}

export function folderBreadcrumbs(
  root: DataFolder,
  folder: DataFolder,
): DataFolder[] {
  if (folder.path === root.path) return [root];

  const crumbs = [root];
  const segments = folder.path.split("/");
  for (let index = 0; index < segments.length; index += 1) {
    const match = findFolder(root, segments.slice(0, index + 1).join("/"));
    if (match) crumbs.push(match);
  }
  return crumbs;
}

function parentPath(path: string): string {
  const segments = path.split("/");
  segments.pop();
  return segments.join("/");
}

function adjacentItem<T>(
  items: readonly T[],
  selectedKey: string,
  offset: -1 | 1,
  key: (item: T) => string,
): T | undefined {
  const index = items.findIndex((item) => key(item) === selectedKey);
  if (index < 0) return undefined;
  return items[index + offset];
}
