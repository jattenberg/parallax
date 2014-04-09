/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.file;

import java.io.File;
import java.io.FileNotFoundException;
import java.lang.reflect.Type;
import java.util.Iterator;

import org.apache.log4j.Logger;

import com.dsi.parallax.pipeline.AbstractExpandingPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Iterators;
import com.google.gson.reflect.TypeToken;

/**
 * FileToLinesPipe is a primitive class for getting contents of File object 
 * into context of string
 *
 * @author Josh Attenberg
 */
public class FileToLinesPipe extends AbstractExpandingPipe<File, String> {
	private static final Logger LOGGER = Logger
			.getLogger(FileToLinesPipe.class);
	private static final long serialVersionUID = 3049605083842704655L;

    /**
     * Class constructor.
     */
	public FileToLinesPipe() {
		super();
	}

    /**
     * The method returns the class's Type "FileToLinesPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<FileToLinesPipe>() {
		}.getType();
	}

	@Override
	protected Iterator<Context<String>> operate(Context<File> context) {
		File file = context.getData();
		try {
			return Iterators.transform(new BufferedReaderIterable(file).iterator(), contextAddingFunction);
		} catch (FileNotFoundException e) {
			LOGGER.error(e.getLocalizedMessage());
			e.printStackTrace();
		}
		// TODO: this will mask errors
		return null;
	}
}
