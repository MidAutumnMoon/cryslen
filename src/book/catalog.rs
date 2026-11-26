//! Book catalog: archive detection, page naming, sibling books, and the
//! natural order shared by listings and sibling walks.

use std::cmp::Ordering;
use std::fs;
use std::fs::DirEntry;
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::path::Path;
use std::path::PathBuf;

use strum::VariantArray;

#[derive(Debug, Clone, Copy, PartialEq, Eq, VariantArray)]
pub enum Direction {
    Forward,
    Backward,
}

impl Direction {
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::Forward => Self::Backward,
            Self::Backward => Self::Forward,
        }
    }
}

/// Non-empty page names in reading order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageNames {
    names: Vec<String>,
    count: NonZeroUsize,
}

impl PageNames {
    pub fn new(names: Vec<String>) -> Option<Self> {
        let count = NonZeroUsize::new(names.len())?;
        Some(Self { names, count })
    }

    pub fn count(&self) -> NonZeroUsize {
        self.count
    }
}

impl Deref for PageNames {
    type Target = [String];

    fn deref(&self) -> &Self::Target {
        &self.names
    }
}

impl<const N: usize> PartialEq<[&str; N]> for PageNames {
    fn eq(&self, other: &[&str; N]) -> bool {
        self.names == other
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageName {
    pub parent_dir: Option<String>,
    pub file: String,
}

pub enum ArchiveKind {
    Zip,
    SevenZip,
}

pub fn archive_kind(path: &Path) -> Option<ArchiveKind> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match extension.as_str() {
        "zip" | "cbz" => Some(ArchiveKind::Zip),
        "7z" | "cb7" => Some(ArchiveKind::SevenZip),
        _ => None,
    }
}

pub fn is_archive(path: &Path) -> bool {
    archive_kind(path).is_some()
}

fn entry_is(entry: &DirEntry, want_dir: bool) -> bool {
    let Ok(file_type) = entry.file_type() else {
        return false;
    };
    #[expect(
        clippy::filetype_is_file,
        reason = "fast path for regular files; symlinks and unknown types fall through to fs::metadata"
    )]
    if file_type.is_file() {
        return !want_dir;
    }
    if file_type.is_dir() {
        return want_dir;
    }
    fs::metadata(entry.path()).is_ok_and(|metadata| {
        if want_dir {
            metadata.is_dir()
        } else {
            metadata.is_file()
        }
    })
}

/// Whether `entry` resolves to a regular file, following symlinks.
pub fn entry_is_file(entry: &DirEntry) -> bool {
    entry_is(entry, false)
}

fn entry_is_dir(entry: &DirEntry) -> bool {
    entry_is(entry, true)
}

/// Resolve relative paths against the working directory and strip `.` components, preserving symlinks.
pub fn normalize_path(path: &Path) -> PathBuf {
    use std::path;

    let absolute =
        path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    absolute.components().collect()
}

/// The next or previous sibling book of the same kind, in natural order;
/// none at either end.
pub fn sibling(
    path: &Path,
    is_archive: bool,
    direction: Direction,
) -> Option<PathBuf> {
    use std::ffi::OsString;

    struct SiblingEntry {
        sort_key: String,
        file_name: OsString,
        path: PathBuf,
    }

    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let current_name = path.file_name()?;
    let mut siblings: Vec<SiblingEntry> = fs::read_dir(parent)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| {
            if is_archive {
                self::is_archive(Path::new(&entry.file_name()))
                    && entry_is_file(entry)
            } else {
                entry_is_dir(entry)
            }
        })
        .map(|entry| {
            let file_name = entry.file_name();
            let sort_key = file_name.to_string_lossy().into_owned();
            SiblingEntry {
                sort_key,
                file_name,
                path: entry.path(),
            }
        })
        .collect();
    siblings.sort_by(|left, right| {
        compare_names(&left.sort_key, &right.sort_key)
    });
    let position = siblings
        .iter()
        .position(|entry| entry.file_name.as_os_str() == current_name)?;
    let neighbor_index = match direction {
        Direction::Forward => position.checked_add(1)?,
        Direction::Backward => position.checked_sub(1)?,
    };
    siblings.get(neighbor_index).map(|entry| entry.path.clone())
}

/// The index of the entry a file argument names, preferring an exact match
/// over a case-insensitive fallback, or the first page when absent.
pub fn landing(entries: &[String], name: &str) -> usize {
    entries
        .iter()
        .position(|entry| entry == name)
        .or_else(|| {
            entries
                .iter()
                .position(|entry| entry.eq_ignore_ascii_case(name))
        })
        .unwrap_or(0)
}

/// The status line's name segment: the basename plus its immediate parent
/// directory. Archives carry the inner path; directories use the source
/// directory's own name.
#[expect(
    clippy::indexing_slicing,
    reason = "the list is non-empty and callers stay below its length; the debug assert restates it"
)]
pub fn page_name(
    names: &[String],
    is_archive: bool,
    directory_name: Option<&str>,
    page: usize,
) -> PageName {
    debug_assert!(page < names.len(), "caller clamps the page index");
    let entry = &names[page];
    let file =
        entry.rsplit(['/', '\\']).next().unwrap_or(entry).to_owned();
    let parent_dir = if is_archive {
        entry.rsplit_once(['/', '\\']).map(|(path, _)| {
            path.rsplit(['/', '\\']).next().unwrap_or(path).to_owned()
        })
    } else {
        directory_name.map(str::to_owned)
    };
    PageName { parent_dir, file }
}

/// Status-line display name for a book before its listing arrives.
pub fn opening_name(path: &Path, land_on: Option<&str>) -> PageName {
    let book_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned());
    if let Some(file) = land_on {
        return PageName {
            parent_dir: book_name,
            file: file.to_owned(),
        };
    }
    let parent_dir = path
        .parent()
        .and_then(Path::file_name)
        .map(|name| name.to_string_lossy().into_owned());
    let file = book_name.unwrap_or_else(|| path.display().to_string());
    PageName { parent_dir, file }
}

/// Natural order over UTF-8 names: ASCII digit runs compare numerically
/// (fewer leading zeros first); remaining text compares by code point.
/// Only identical names compare equal.
pub fn compare_names(left: &str, right: &str) -> Ordering {
    let (mut left, mut right) = (left.as_bytes(), right.as_bytes());
    loop {
        match (left.first(), right.first()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(&left_byte), Some(&right_byte)) => {
                if left_byte.is_ascii_digit()
                    && right_byte.is_ascii_digit()
                {
                    let (left_run, left_rest) = leading_digits(left);
                    let (right_run, right_rest) = leading_digits(right);
                    let order = compare_digit_runs(left_run, right_run);
                    if order != Ordering::Equal {
                        return order;
                    }
                    left = left_rest;
                    right = right_rest;
                } else {
                    let order = left_byte.cmp(&right_byte);
                    if order == Ordering::Equal {
                        left = left.split_at(1).1;
                        right = right.split_at(1).1;
                    } else {
                        return order;
                    }
                }
            }
        }
    }
}

fn leading_digits(bytes: &[u8]) -> (&[u8], &[u8]) {
    let end = bytes
        .iter()
        .position(|byte| !byte.is_ascii_digit())
        .unwrap_or(bytes.len());
    bytes.split_at(end)
}

fn strip_leading_zeros(digits: &[u8]) -> &[u8] {
    let start = digits
        .iter()
        .position(|&byte| byte != b'0')
        .unwrap_or(digits.len());
    digits.split_at(start).1
}

/// By value, then fewer leading zeros.
fn compare_digit_runs(left: &[u8], right: &[u8]) -> Ordering {
    let (left_value, right_value) =
        (strip_leading_zeros(left), strip_leading_zeros(right));
    let (left_zeros, right_zeros) = (
        left.len() - left_value.len(),
        right.len() - right_value.len(),
    );
    left_value
        .len()
        .cmp(&right_value.len())
        .then_with(|| left_value.cmp(right_value))
        .then_with(|| left_zeros.cmp(&right_zeros))
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "a panic in test code is inconsequential"
)]
mod tests {
    use std::fs::File;

    use tempfile::TempDir;
    use zip::ZipWriter;

    use super::*;
    use crate::testkit;

    #[test]
    fn page_name_resolves_archive_and_directory_entries() {
        let names = vec![
            "p1.png".to_owned(),
            "b/010.png".to_owned(),
            "b/deep/005.png".to_owned(),
        ];
        assert_eq!(
            page_name(&names, true, None, 0),
            PageName {
                parent_dir: None,
                file: "p1.png".into(),
            }
        );
        assert_eq!(
            page_name(&names, true, None, 1),
            PageName {
                parent_dir: Some("b".into()),
                file: "010.png".into(),
            }
        );
        assert_eq!(
            page_name(&names, true, None, 2),
            PageName {
                parent_dir: Some("deep".into()),
                file: "005.png".into(),
            }
        );
        assert_eq!(
            page_name(&names, false, Some("vol"), 0),
            PageName {
                parent_dir: Some("vol".into()),
                file: "p1.png".into(),
            }
        );
    }

    #[test]
    fn landing_resolves_the_image_argument() {
        let entries: Vec<String> =
            ["cover.png", "Page2.png", "page2.png", "p3.png"]
                .into_iter()
                .map(String::from)
                .collect();
        assert_eq!(landing(&entries, "Page2.png"), 1);
        assert_eq!(landing(&entries, "page2.png"), 2);
        assert_eq!(landing(&entries, "page2.PNG"), 1);
        assert_eq!(landing(&entries, "missing.png"), 0);
    }

    #[cfg(unix)]
    #[test]
    fn sibling_follows_symlinks_and_skips_invalid_entries() {
        use std::os::unix::fs as unix_fs;

        let root = TempDir::new().expect("create temp dir");
        let books = root.path().join("books");
        let targets = root.path().join("targets");
        fs::create_dir_all(&books).expect("mkdir books");
        fs::create_dir_all(&targets).expect("mkdir targets");

        fs::create_dir_all(books.join("1-dir")).expect("1-dir");
        unix_fs::symlink(targets.join("missing"), books.join("2-broken"))
            .expect("broken dir symlink");
        fs::create_dir_all(targets.join("real-dir")).expect("real dir");
        unix_fs::symlink(targets.join("real-dir"), books.join("3-link"))
            .expect("symlink dir");
        assert_eq!(
            sibling(&books.join("1-dir"), false, Direction::Forward),
            Some(books.join("3-link"))
        );

        fs::write(books.join("1-book.cbz"), b"z").expect("1-book.cbz");
        unix_fs::symlink(
            targets.join("missing.cbz"),
            books.join("2-broken.cbz"),
        )
        .expect("broken archive symlink");
        unix_fs::symlink(
            targets.join("real-dir"),
            books.join("3-dir.cbz"),
        )
        .expect("archive symlink pointing at dir");
        fs::write(targets.join("real-book.cbz"), b"z")
            .expect("real-book.cbz");
        unix_fs::symlink(
            targets.join("real-book.cbz"),
            books.join("4-link.cbz"),
        )
        .expect("symlink archive");
        assert_eq!(
            sibling(&books.join("1-book.cbz"), true, Direction::Forward),
            Some(books.join("4-link.cbz"))
        );
    }

    #[test]
    fn natural_order_compares_digit_runs_by_value() {
        let mut names = vec![
            "9.png".to_owned(),
            "010.png".to_owned(),
            "1.png".to_owned(),
        ];
        names.sort_by(|left, right| compare_names(left, right));
        assert_eq!(names, ["1.png", "9.png", "010.png"]);
        let mut padded =
            vec!["page10.png".to_owned(), "page09.png".to_owned()];
        padded.sort_by(|left, right| compare_names(left, right));
        assert_eq!(padded, ["page09.png", "page10.png"]);
        assert_eq!(compare_names("10.png", "010.png"), Ordering::Less);
        assert_eq!(compare_names("010.png", "10.png"), Ordering::Greater);
        assert_eq!(compare_names("10.png", "10.png"), Ordering::Equal);
        assert_eq!(
            compare_names("Page2.png", "page3.png"),
            Ordering::Less
        );
        assert_eq!(
            compare_names("page 2.png", "page2.png"),
            Ordering::Less
        );
        assert_eq!(compare_names("page.png", "page2.png"), Ordering::Less);
        assert_eq!(compare_names("b", "a"), Ordering::Greater);
        assert_eq!(compare_names("", ""), Ordering::Equal);
        assert_eq!(compare_names("", "a"), Ordering::Less);
    }

    #[test]
    fn natural_order_handles_non_ascii_prefixes() {
        assert_eq!(
            compare_names("漫画-2.jpg", "漫画-10.jpg"),
            Ordering::Less
        );
        assert_eq!(
            compare_names("第1話.cbz", "第10話.cbz"),
            Ordering::Less
        );
        assert_eq!(compare_names("阿", "白"), Ordering::Greater);
        assert_eq!(
            compare_names("06-漫画.jpg", "漫画-06.jpg"),
            Ordering::Less
        );
    }

    #[test]
    fn natural_order_is_total() {
        let names = [
            "",
            "0",
            "00",
            "1",
            "01",
            "9",
            "10",
            "010",
            "A",
            "a",
            "a0",
            "a1",
            "a01",
            "b",
            "page 2",
            "page2",
            "z",
            "é",
            "第1話",
            "第10話",
            "漫画-2",
            "漫画-10",
            "阿",
            "白",
        ];
        for &left in &names {
            for &right in &names {
                let forward = compare_names(left, right);
                assert_eq!(
                    forward,
                    compare_names(right, left).reverse(),
                    "{left:?} vs {right:?}"
                );
                assert_eq!(
                    forward == Ordering::Equal,
                    left == right,
                    "{left:?} vs {right:?}"
                );
            }
        }
        for &left in &names {
            for &middle in &names {
                if compare_names(left, middle) == Ordering::Greater {
                    continue;
                }
                for &right in &names {
                    if compare_names(middle, right) == Ordering::Greater {
                        continue;
                    }
                    assert_ne!(
                        compare_names(left, right),
                        Ordering::Greater,
                        "{left:?} <= {middle:?} <= {right:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn sibling_books_step_between_their_own_kind() {
        let parent = TempDir::new().expect("create temp dir");
        let volume = |name: &str| {
            let dir = parent.path().join(name);
            fs::create_dir_all(&dir).expect("volume dir creates");
            fs::write(
                dir.join("p.png"),
                testkit::tiny_png(1, 1, [0, 0, 0, 255]),
            )
            .expect("volume page writes");
            dir
        };
        let first = volume("v1");
        let middle = volume("v2");
        let last = volume("v10");
        fs::write(parent.path().join("notes.txt"), b"not a book")
            .expect("plain file writes");

        assert_eq!(
            sibling(&middle, false, Direction::Forward),
            Some(last),
            "forward follows natural order"
        );
        assert_eq!(
            sibling(&middle, false, Direction::Backward),
            Some(first.clone()),
            "backward follows natural order"
        );
        assert_eq!(
            sibling(&first, false, Direction::Backward),
            None,
            "the shelf's start has no earlier book"
        );

        let shelf = TempDir::new().expect("create temp dir");
        let write_cbz = |name: &str| {
            use std::io::Write as _;

            use zip::write::SimpleFileOptions;

            let path = shelf.path().join(name);
            let file = File::create(&path).expect("zip creates");
            let mut zip = ZipWriter::new(file);
            zip.start_file("p.png", SimpleFileOptions::default())
                .expect("entry starts");
            zip.write_all(&testkit::tiny_png(1, 1, [0, 0, 0, 255]))
                .expect("entry writes");
            zip.finish().expect("zip finishes");
            path
        };
        let before = write_cbz("a.cbz");
        let after = write_cbz("b.cbz");
        fs::write(shelf.path().join("notes.txt"), b"not a book")
            .expect("plain file writes");
        assert_eq!(
            sibling(&before, true, Direction::Forward),
            Some(after),
            "archives step between archives"
        );
        assert_eq!(
            sibling(&before, true, Direction::Backward),
            None,
            "the first archive has no earlier book"
        );
    }
}
