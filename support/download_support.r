if (!requireNamespace("Hmisc", quietly = TRUE)) {
	install.packages("Hmisc")
}

suppressPackageStartupMessages(library(Hmisc))

# NOTE:
# `support` は `data(support)` で読める同梱データではなく、
# Hmisc のデータレポから取得する想定（getHdata）です。
options(timeout = max(60L, getOption("timeout", 60L)))

tryCatch(
	{
		Hmisc::getHdata("support")
	},
	error = function(e) {
		message("Hmisc::getHdata('support') に失敗しました: ", conditionMessage(e))
		message("ネットワーク接続を確認するか、where 引数でミラーを指定してください。")
		message("例: Hmisc::getHdata('support', where='https://hbiostat.org/data/repo')")
		stop(e)
	}
)

script_file <- grep("^--file=", commandArgs(), value = TRUE)
script_dir <- if (length(script_file) >= 1) {
	dirname(normalizePath(sub("^--file=", "", script_file[[1]]), mustWork = FALSE))
} else {
	getwd()
}

out_csv <- file.path(script_dir, "support.csv")
utils::write.csv(support, file = out_csv, row.names = FALSE, fileEncoding = "UTF-8")
message("Wrote: ", out_csv)