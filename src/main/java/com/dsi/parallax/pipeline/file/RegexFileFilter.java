/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.file;

import org.apache.commons.io.filefilter.IOFileFilter;

import java.io.File;
import java.util.regex.Pattern;

/**
 * RegexFileFilter implements IOFileFilter class that filters file base on regular expression of file path and
 * regular expression of  file name
 *
 * @author Josh Attenberg
 */
public class RegexFileFilter implements IOFileFilter {

    private Pattern pathRegex;
    private Pattern filenameRegex;

    /**
     * Class constructor specifying Pattern regular expression of file name to create.
     * @param filenameRegex regular expression of file name
     */
    public RegexFileFilter(Pattern filenameRegex) {
        this(null, filenameRegex);
    }

    /**
     * Class constructor specifying Pattern regular expression of file name and file path to create.
     * @param pathRegex regular expression of file path
     * @param filenameRegex regular expression of file name
     */
    public RegexFileFilter(Pattern pathRegex, Pattern filenameRegex) {
        this.filenameRegex = filenameRegex;
        this.pathRegex = pathRegex;
    }

    /**
     * Class constructor specifying String regular expression of file name and file path to create     *
     * @param pathRegex String regular expression
     * @param filenameRegex String regular expression
     */
    public RegexFileFilter(String pathRegex, String filenameRegex) {
        this(pathRegex==null?null:Pattern.compile(pathRegex), filenameRegex==null?null:Pattern.compile(filenameRegex));
    }

    /**
     *  Class constructor specifying String regular expression of file name to create
     * @param filenameRegex String regular expression
     */
    public RegexFileFilter(String filenameRegex) {
        this(null, filenameRegex);
    }

    /**
     * The method checks if inputted file is accepted
     * @param pathname file
     * @return boolean
     */
    @Override
    public boolean accept(File pathname) {
        boolean accept = (pathRegex == null || pathRegex.matcher(pathname.getAbsolutePath()).matches()) &&
                    ( filenameRegex == null || filenameRegex.matcher(pathname.getName()).matches());
        return accept;
    }

    /**
     * The method checks if inputted file is accepted
     * @param file file
     * @param name file name
     * @return boolean
     */
    @Override
    public boolean accept(File file, String name) {
        File tmpfile = new File(file, name);
        return accept(tmpfile);
    }
}
