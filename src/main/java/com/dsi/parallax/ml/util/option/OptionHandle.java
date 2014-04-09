package com.dsi.parallax.ml.util.option;

/**
 * enum used for describing individual options
 * replacing short and long names, multiple hashmaps in Configuration
 * @author jattenberg
 *
 */
public enum OptionHandle {
	NONE {
		@Override
		public String shortName() {
			return "none";
		}

		@Override
		public String longName() {
			return "none";
		}
	};
	
	public abstract String shortName();
	public abstract String longName();
}
