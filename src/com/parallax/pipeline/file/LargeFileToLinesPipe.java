/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.file;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.Iterator;

import org.apache.log4j.Logger;

import com.google.common.collect.Iterators;
import com.google.gson.reflect.TypeToken;
import com.parallax.pipeline.AbstractExpandingPipe;
import com.parallax.pipeline.Context;

/**
 * Gets content of a file using memory mapping.
 *
 * @author Rahul Ratnakar 
 */
public class LargeFileToLinesPipe extends AbstractExpandingPipe<File, String> {
	private static final Logger LOGGER = Logger
			.getLogger(FileToLinesPipe.class);
	private static final long serialVersionUID = 3049605083842704655L;

    /**
     * Class constructor.
     */
	public LargeFileToLinesPipe() {
		super();
	}

    /**
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<LargeFileToLinesPipe>() {
		}.getType();
	}

	@Override
	protected Iterator<Context<String>> operate(Context<File> context) {
		File file = context.getData();
		try {
			return Iterators.transform(new MemoryMappedFileReaderIterable(file).iterator(), contextAddingFunction);
		} catch (IOException e) {
			LOGGER.error(e.getLocalizedMessage());
			e.printStackTrace();
		}
		return null;
	}
}
