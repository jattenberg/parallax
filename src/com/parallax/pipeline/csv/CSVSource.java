package com.parallax.pipeline.csv;

import java.io.File;
import java.lang.reflect.Type;
import java.util.Iterator;

import com.google.gson.reflect.TypeToken;
import com.parallax.ml.util.csv.CSV;
import com.parallax.ml.vector.LinearVector;
import com.parallax.pipeline.Context;
import com.parallax.pipeline.FileSource;
import com.parallax.pipeline.Pipeline;
import com.parallax.pipeline.Source;
import com.parallax.pipeline.file.FileToLinesPipe;
import com.parallax.pipeline.text.StringSplitPipe;

public class CSVSource implements Source<LinearVector> {

	private static final long serialVersionUID = 2560396989298236707L;
	private final Pipeline<File, LinearVector> pipeline;

	public CSVSource(File file, CSV format, char delim) {
		pipeline = Pipeline.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new StringSplitPipe(delim))
				.addPipe(new CSVPipe(format));
	}

	@Override
	public Iterator<Context<LinearVector>> provideData() {
		return pipeline.process();
	}

	@Override
	public Type getType() {
		return new TypeToken<CSVSource>() {
		}.getType();
	}

}
