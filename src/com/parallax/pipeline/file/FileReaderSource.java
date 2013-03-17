package com.parallax.pipeline.file;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.Reader;
import java.lang.reflect.Type;
import java.util.Iterator;
import java.util.List;

import com.google.common.collect.Lists;
import com.google.gson.reflect.TypeToken;
import com.parallax.pipeline.Context;
import com.parallax.pipeline.GenericContextIterator;
import com.parallax.pipeline.Source;

public class FileReaderSource implements Source<BufferedReader> {

	private static final long serialVersionUID = 836189510061543077L;
	private List<BufferedReader> readers;
	private Iterator<Context<BufferedReader>> it;

	public FileReaderSource(Reader reader) {
		this(new BufferedReader(reader));
	}

	public FileReaderSource(BufferedReader reader) {
		readers = Lists.newArrayList(reader);
		it = new GenericContextIterator<BufferedReader>(readers.iterator());
	}

	public FileReaderSource(File file) throws FileNotFoundException {
		this(new BufferedReader(new FileReader(file)));
	}

	@Override
	public Iterator<Context<BufferedReader>> provideData() {
		return it;
	}

	@Override
	public Type getType() {
		return new TypeToken<FileReaderSource>() {
		}.getType();
	}

}
