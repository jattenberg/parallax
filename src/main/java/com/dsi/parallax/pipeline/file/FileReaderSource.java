package com.dsi.parallax.pipeline.file;

import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.GenericContextIterator;
import com.dsi.parallax.pipeline.Source;
import com.google.common.collect.Lists;
import com.google.gson.reflect.TypeToken;

import java.io.*;
import java.lang.reflect.Type;
import java.util.Iterator;
import java.util.List;

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
