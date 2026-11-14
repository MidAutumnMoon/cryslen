//! Book catalog: archive detection, page naming, sibling books, and the
//! natural order shared by listings and sibling walks.

use std::cmp::Ordering;
use std::ffi::OsString;
use std::fs;
use std::fs::DirEntry;
use std::path;
use std::path::Path;
use std::path::PathBuf;

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
    let ext = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "zip" | "cbz" => Some(ArchiveKind::Zip),
        "7z" | "cb7" => Some(ArchiveKind::SevenZip),
        _ => None,
    }
}

pub fn is_archive(path: &Path) -> bool {
    archive_kind(path).is_some()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DirTarget {
    File,
    Dir,
}

fn entry_target(entry: &DirEntry) -> Option<DirTarget> {
    let file_type = entry.file_type().ok()?;
    #[expect(
        clippy::filetype_is_file,
        reason = "fast path for regular files; symlinks and unknown types fall through to fs::metadata"
    )]
    if file_type.is_file() {
        return Some(DirTarget::File);
    }
    if file_type.is_dir() {
        return Some(DirTarget::Dir);
    }
    let metadata = fs::metadata(entry.path()).ok()?;
    if metadata.is_dir() {
        Some(DirTarget::Dir)
    } else if metadata.is_file() {
        Some(DirTarget::File)
    } else {
        None
    }
}

/// Whether `entry` resolves to a regular file, following symlinks.
pub fn entry_is_file(entry: &DirEntry) -> bool {
    entry_target(entry) == Some(DirTarget::File)
}

/// Resolve relative paths against the working directory and strip `.` components, preserving symlinks.
pub fn normalize_path(path: &Path) -> PathBuf {
    let absolute =
        path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    absolute.components().collect()
}

/// The next or previous sibling book of the same kind, in natural order;
/// none at either end.
pub fn sibling(
    path: &Path,
    archive: bool,
    forward: bool,
) -> Option<PathBuf> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let current = path.file_name()?;
    let mut siblings: Vec<(String, OsString, PathBuf)> = fs::read_dir(parent)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| {
            if archive {
                is_archive(Path::new(&entry.file_name()))
                    && entry_is_file(entry)
            } else {
                entry_target(entry) == Some(DirTarget::Dir)
            }
        })
        .map(|entry| {
            let file_name = entry.file_name();
            let sort_key = file_name.to_string_lossy().into_owned();
            (sort_key, file_name, entry.path())
        })
        .collect();
    siblings.sort_by(|left, right| compare_names(&left.0, &right.0));
    let position = siblings
        .iter()
        .position(|(_, file_name, _)| file_name.as_os_str() == current)?;
    let next = if forward {
        position.checked_add(1)?
    } else {
        position.checked_sub(1)?
    };
    siblings.get(next).map(|(_, _, sibling)| sibling.clone())
}

/// The index of the entry a file argument names, or the first page when
/// the name is absent.
pub fn landing<'entries>(
    mut entries: impl Iterator<Item = &'entries str>,
    name: &str,
) -> usize {
    entries
        .position(|entry| entry.eq_ignore_ascii_case(name))
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
    archive: bool,
    dir_name: Option<&str>,
    index: usize,
) -> PageName {
    debug_assert!(index < names.len(), "caller clamps the page index");
    let entry = &names[index];
    let file = entry.rsplit('/').next().unwrap_or(entry).to_owned();
    let parent_dir = if archive {
        entry.rsplit_once('/').map(|(path, _)| {
            path.rsplit('/').next().unwrap_or(path).to_owned()
        })
    } else {
        dir_name.map(str::to_owned)
    };
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
    use super::*;

    #[test]
    fn landing_resolves_the_image_argument() {
        let entries = ["cover.png", "Page2.png", "p3.png"];
        let names = || entries.iter().copied();
        assert_eq!(landing(names(), "Page2.png"), 1);
        assert_eq!(landing(names(), "page2.PNG"), 1);
        assert_eq!(landing(names(), "missing.png"), 0);
    }

    #[cfg(unix)]
    #[test]
    fn sibling_follows_symlinks_and_skips_invalid_entries() {
        use std::os::unix::fs::symlink;

        use crate::testkit::TempDir;

        let root = TempDir::new("catalog-siblings");
        let books = root.path().join("books");
        let targets = root.path().join("targets");
        fs::create_dir_all(&books).expect("mkdir books");
        fs::create_dir_all(&targets).expect("mkdir targets");

        fs::create_dir_all(books.join("1-dir")).expect("1-dir");
        symlink(targets.join("missing"), books.join("2-broken"))
            .expect("broken dir symlink");
        fs::create_dir_all(targets.join("real-dir")).expect("real dir");
        symlink(targets.join("real-dir"), books.join("3-link"))
            .expect("symlink dir");
        assert_eq!(
            sibling(&books.join("1-dir"), false, true),
            Some(books.join("3-link"))
        );

        fs::write(books.join("1-book.cbz"), b"z").expect("1-book.cbz");
        symlink(targets.join("missing.cbz"), books.join("2-broken.cbz"))
            .expect("broken archive symlink");
        symlink(targets.join("real-dir"), books.join("3-dir.cbz"))
            .expect("archive symlink pointing at dir");
        fs::write(targets.join("real-book.cbz"), b"z").expect("real-book.cbz");
        symlink(targets.join("real-book.cbz"), books.join("4-link.cbz"))
            .expect("symlink archive");
        assert_eq!(
            sibling(&books.join("1-book.cbz"), true, true),
            Some(books.join("4-link.cbz"))
        );
    }
}
