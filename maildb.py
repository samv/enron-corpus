"""Batch script to load the entire enron corpus, construct a Trie in RAM as
well as collect counters and write out the index files when it is done."""

import email.parser
import os
import re
import struct
import zlib

from BitVector import BitVector


PERTINENT_HEADERS = set((
    "From", "To", "Cc", "Bcc", "Subject", "Date",
))
IGNORE_HEADERS = set((
    "Received-From",
))
seen_other_headers = set()


class MailDB(object):
    def __init__(self, root_path):
        self.root_path = root_path
	self.parser = email.parser.Parser()
	self.prefix_db = PrefixIndex(root_path)

    def files(self, pattern=None):
        for path, dirs, files in os.walk(self.root_path):
	    rel_path = os.path.relpath(path, self.root_path)
	    if rel_path == "":  # built indexes will live at root
		continue
            for file in files:
		if pattern is None or re.search(
		    pattern, os.path.join(rel_path, file)
		):
		    yield rel_path, file

    def index_files(self, pattern):
	for rel_path, file in self.files(pattern):
	    self.index_file(rel_path, file)

    def index_file(self, rel_path, filename):
        filename = os.path.join(self.root_path, rel_path, filename)
	offset = self.prefix_db.number_file(rel_path, filename)
        fh = open(filename, "r", 1)
	message = self.parser.parse(fh)
	fh.close()
	self.index_message(message, offset)
	self.prefix_db.progress(rel_path, filename)

    def index_message(self, message, offset):
	all_words = set()
	for header, value in message.items():
	    if header in IGNORE_HEADERS:
		pass
	    else:
		if header not in PERTINENT_HEADERS and \
			header not in seen_other_headers:
		    seen_other_headers.add(header)
		    print "Warning: header %s not triaged" % header
	        words = re.findall(r"(\w+)", value, re.U)
		all_words.update(set(x.lower() for x in words))
	for part in message.walk():
	    words = re.findall(r"(\w+)", str(part), re.U)
	    all_words.update(set(x.lower() for x in words))
	self.prefix_db.mark_tags(offset, all_words)

    def commit(self):
	self.prefix_db.write_filenames()
	self.prefix_db.write_index()


PREFIX_IDX = "prefixidx.dat"
FILENAMES_IDX = "filenameidx.dat"

ZERO = "\0" * 4


class PrefixIndex(object):
    """Implements a Trie index based on prefixes for searching the Enron
    corpus."""
    def __init__(self, db_path):
	self.db_path = db_path
	self.filenames = []
	self.trie = {"_c": 0}  # keys: next character
	self.bitmap_size = 524288  # 2 ** 19
	self.bitmaps = 0
	self.trie_size = 0

    def progress(self, rel_path, filename):
	print "Indexed {rp}/{fn}, trie size = {x}, bitmaps = {n} (total: {s}MiB)".format(
	    rp=rel_path,
	    fn=filename,
	    x=self.trie_size,
	    n=self.bitmaps,
	    s=(self.bitmaps >> 1),
	)

    def number_file(self, rel_path, filename):
        self.filenames += os.path.join(rel_path, filename)
	return len(self.filenames)

    def write_filenames(self):
	"""Write out the filenames index.
	Header: int4: total filenames
                int4[n]: offsets to compressed filenames
	Data: 32 concatenated zlib streams
	"""
	idx = open(
	    os.path.join(self.db_path, FILENAMES_IDX), "w", 4096
        )
	num_filenames = len(self.filenames)
	idx.write(struct.pack(">l", num_filenames))
	num_chunks = (num_filenames + 1023) >> 10
	idx.write(ZERO * num_chunks)

	for chunk in xrange(0, num_chunks):
	    offs = idx.tell()
	    # update the lookup table
	    idx.seek(4 * (chunk + 1))
	    idx.write(struct.pack(">l", offs))
	    idx.seek(0, 2)
	    filenames = "\0".join(self.filenames[chunk<<10:(chunk+1)<<10])
	    zfilenames = zlib.compress(filenames)
	    idx.write(zfilenames)
	    # align on 4-byte boundaries, why not
	    if len(zfilenames) & 3:
	        idx.write("\0" * (4 - (len(zfilenames) & 3)))

        idx.close()

    def mark_tag(self, offset, tag, seen_set):
	"""Updates the trie with a single tag"""
	node = self.trie
        i = 0
	while len(tag) > i:
            letter = tag[i]
	    i += 1
	    substr = tag[0:i]
	    if letter not in node:
		self.trie_size += 1
		node[letter] = {"_c": 1}
	    elif substr not in seen_set:
		node[letter]["_c"] += 1
	    node = node[letter]
	    seen_set.add(substr)
	if "_b" not in node:
	    self.bitmaps += 1
	    node["_b"] = BitVector(size = self.bitmap_size)
	node["_b"][offset] = 1

    def mark_tags(self, offset, tags):
	"""Given an offset and a list of words/tags, mark the bits at
	offset 'offset' as 1 for each tag, and update the counters for
	all substrings."""
	seen_set = set()
	self.trie["_c"] += 1
	for tag in tags:
	    self.mark_tag(offset, tag, seen_set)

    def write_index(self):
	"""Write out the actual search index.

	Format:
 	  Page Header:
	    int4 - count of matches for this item
	    int4 - number of entries in this page
	    int4[N] - ordered codepoints of entries
	    int4[N] - file offsets to next page
          Page Data:
            int4 - size of bitmap
            data[X] - bitmap data (zlib compressed)
        """
	sidx = open(
	    os.path.join(self.db_path, PREFIX_IDX), "w", 4096
        )
	num_filenames = len(self.filenames)
        pointers_todo = {}  # key => .tell()
        trie_todo = [("", self.trie)]
	count = 0
	import pdb; pdb.set_trace()
        while len(trie_todo):
	    prefix, trie = trie_todo.pop()
	    # update the pointer which points to this node
	    if len(prefix):
		pos = sidx.tell()
		sidx.seek(pointers_todo[prefix], 0)
		sidx.write(struct.pack(">l", pos))
		sidx.seek(0, 2)
		del pointers_todo[prefix]
	    # write match count
	    sidx.write(struct.pack(">l", trie["_c"]))
	    children = set(trie.keys())
	    children.remove("_c")
	    if "_b" in children:
		children.remove("_b")
            children = sorted(children)
	    sidx.write(struct.pack(">l", len(children)))
	    for child in children:
		sidx.write(struct.pack(">l", ord(child)))
		pointers_todo[prefix + child] = sidx.tell()
		sidx.write(ZERO)
		trie_todo.append((prefix + child, trie[child]))
	    if "_b" in trie:
		print "(gTFBV)",
		bitmap = trie["_b"].getTextFromBitVector()
		print "(zlib)",
	    	zbitmap = zlib.compress(bitmap)
		sidx.write(struct.pack(">l", len(zbitmap)))
		sidx.write(zbitmap)
	        # align on 4-byte boundaries, why not
	        if len(zbitmap) & 3:
	            sidx.write("\0" * (4 - (len(zbitmap) & 3)))
	    else:
		sidx.write(ZERO)
	    count += 1
	    print "\rWrote [%d/%d] nodes..." % (count, self.trie_size),


if __name__ == '__main__':
    prefix_idx = MailDB("data/skilling-j")
    prefix_idx.index_files(r"inbox/83")
    prefix_idx.commit()
