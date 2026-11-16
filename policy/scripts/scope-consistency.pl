#!/usr/bin/env perl
# policy/scripts/scope-consistency.pl — mechanical layer of
# EXP.SCOPE_EXCLUSION_CONSISTENCY. Called by policy/lint.sh.
#
#   A (locator)  every negative-scope statement (not supported, excluded,
#                for X only / no X claim, abstains, does not apply, outside the
#                scope, not evaluated) that names an object, paired with the
#                table rows and prose sentences elsewhere that report a number
#                for the same object. A pair is a question for the judgment
#                layer, never a verdict: most pairs are consistent.
#   B1           the same definite population ("the 1,200 attacked
#                configurations", "k of the n <noun>") written with different
#                denominators in different files.
#   B2           a percentage printed next to k/n or "k of n" that does not
#                equal 100k/n at the precision shown.
#   B3           a prose "k of the n" with a large denominator whose table cells
#                with the same denominator never show numerator k.
#
# Not covered (stated so silence is not read as clean): "--" / N/A cells,
# implicit inclusion ("all ten schemes" without naming the excluded one), and
# prose totals against table column sums.
#
# Usage: perl scope-consistency.pl <target-root> <list-file>
#   list-file lines: "<source path>\t<lint view path>". The view is lint.sh's
#   comment-blanked copy with hard-wrapped prose joined onto its first line, so
#   line numbers are the source's.
# Output: "F\t<path>:<line>: <detail>" findings, "I\t<text>" info lines.
# Env: SCOPE_TOPN (pairs shown per statement, default 5, 0 = all),
#      SCOPE_B3_MIN_DEN (default 1000).

use strict;
use warnings;

my ($ROOT, $LIST) = @ARGV;
die "usage: $0 <root> <list>\n" unless defined $LIST;
$ROOT =~ s{/+$}{};
my $TOPN    = $ENV{SCOPE_TOPN} // 5;
my $MIN_DEN = $ENV{SCOPE_B3_MIN_DEN} // 1000;

my $NUM = qr/\d{1,3}(?:,\d{3})+|\d+/;
my $NEG = qr/\b(?:not\s+supported|unsupported|exclud(?:e|es|ed|ing)|for\s+[\w-]+(?:\s+[\w-]+)?\s+only|no\s+[\w-]+\s+claims?|abstain(?:s|ed|ing)?|do(?:es)?\s+not\s+apply|outside\s+(?:the|our)\s+scope|not\s+evaluated|not\s+applicable)\b/i;
my $VERB = qr/\b(?:scor(?:e|es|ed|ing)|accept(?:s|ed|ing|ance)?|reject(?:s|ed|ing|ion)?|abstain(?:s|ed)?|quer(?:y|ies|ied))\b/i;

my %STOP = map { $_ => 1 } qw(
  about above after again against among because before being below between
  cannot could does doing during each either every first from further given
  having however into itself least might more most never only other our ours
  over same should since some such than that their them then there these they
  this those through under until upon very were what when where whether which
  while whose with within without would table tables section figure appendix
  report reports reported shows shown using based therefore thus hence also
  claim claims supported unsupported excluded exclude excludes only apply
  applies outside scope evaluated applicable abstain abstains here both
);
my %CUT = map { $_ => 1 } qw(
  of in on at by for from to with under after before during across per than
  that which who whose where when while is are was were be been has have had
  do does did and or but nor as so if we our they their it its this these those
  each every both remain remains remained come comes came are pass passes
  reach reaches show shows yield yields
);
my %UNIT = map { $_ => 1 } qw(
  second minute hour day week month year ms gb mb kb tb percent time page
  point pt x bit byte
);

sub stem { my $w = lc shift; $w =~ s/ies$/y/ or $w =~ s/(?<=[a-z]{3})(?<!s)s$//; $w }

# LaTeX → readable text with numbers kept. Formatting wrappers are unwrapped,
# reference/colour plumbing removed, numeric math unwrapped, other math dropped.
sub clean {
  my ($s, $italic) = @_;
  $s =~ s/\\_/_/g;
  $s =~ s/(\d)\{,\}(\d)/$1,$2/g;
  $s =~ s/(\d)\\[,;!]\s*(\d)/$1$2/g;
  $s =~ s/\\(?:cite[a-zA-Z]*|[cC]ref|ref|autoref|eqref|pageref|label|apx|tnote|url|href|input|include|includegraphics|cellcolor|rowcolor|textcolor|color|setlength|hline|cline|multirow)\*?(?:\[[^\]]*\])*(?:\{[^{}]*\})+/ /g;
  $s =~ s/\b(?:blue|red|pink|yellow|green|gray|grey|cyan|orange|black|white)!\d+/ /g;
  if ($italic) {
    while ($s =~ /\\(?:textit|texttt|emph)\{([^{}]{2,40})\}/g) { $italic->{$1} = 1 }
  }
  for (1 .. 3) { $s =~ s/\\[a-zA-Z]+\*?\{([^{}]*)\}/$1/g }
  $s =~ s/\$\s*([\d.,\/\s]+?)\s*\$/$1/g;
  $s =~ s/\\\(\s*([\d.,\/\s]+?)\s*\\\)/$1/g;
  $s =~ s/\\\(.*?\\\)/ /g;
  my $dollars = () = $s =~ /(?<!\\)\$/g;
  $s =~ s/(?<!\\)\$[^\$]*(?<!\\)\$/ /g if $dollars % 2 == 0;
  $s =~ s/\\%/%/g;
  $s =~ s/\\[a-zA-Z]+\*?/ /g;
  $s =~ s/[~{}]/ /g;
  $s =~ s/\s+/ /g;
  $s =~ s/^\s+|\s+$//g;
  return $s;
}

sub sentences { my $t = shift; return grep { /\S/ } split /(?<=[.!?])\s+(?=[A-Z(\[])/, $t }
sub num { my $n = shift; $n =~ s/,//g; return $n }
sub fmt { my $n = shift; 1 while $n =~ s/^(\d+)(\d{3})/$1,$2/; return $n }
sub snip { my ($s, $len) = @_; $len //= 90; return length($s) > $len ? substr($s, 0, $len - 3) . '...' : $s }
sub has_num { my $s = shift; return $s =~ /(?<![\w.\-\/^])\d[\d,]*(?:\.\d+)?(?:\s*\/\s*\d[\d,]*)?(?![\w])/ }

# ─── Load files ──────────────────────────────────────────────────────────────
my @F;
open(my $lf, '<', $LIST) or die "cannot read $LIST\n";
while (my $pair = <$lf>) {
  chomp $pair;
  my ($src, $view) = split /\t/, $pair;
  next unless defined $view && open(my $fh, '<', $view);
  my @raw = <$fh>; close $fh; chomp @raw;
  (my $rel = $src) =~ s{^\Q$ROOT\E/}{};
  (my $base = lc $rel) =~ s{.*/}{};
  $base =~ s/\.tex$//; $base =~ s/[_\-.]+/ /g;
  my $f = { src => $src, rel => $rel, base => $base, raw => \@raw, L => [] };
  my ($tab, $skip, $caption) = (0, 0, '');
  for my $i (0 .. $#raw) {
    my $r = $raw[$i];
    $tab++  if $r =~ /\\begin\{(?:tabular|tabularx|tabular\*|longtable|array)\}/;
    $skip++ if $r =~ /\\begin\{(?:equation|align|gather|multline|algorithm|algorithmic|lstlisting|verbatim|tikzpicture)\*?\}/;
    my %it;
    my $c = clean($r, \%it);
    my $row = ($tab > 0 && $r =~ /(?<!\\)&/) ? 1 : 0;
    $caption .= ' ' . lc $c if $r =~ /\\caption/;
    $f->{L}[$i] = { c => $c, row => $row, skip => ($skip > 0 ? 1 : 0), it => \%it };
    $tab--  if $r =~ /\\end\{(?:tabular|tabularx|tabular\*|longtable|array)\}/ && $tab > 0;
    $skip-- if $r =~ /\\end\{(?:equation|align|gather|multline|algorithm|algorithmic|lstlisting|verbatim|tikzpicture)\*?\}/ && $skip > 0;
  }
  $f->{caption} = $caption;
  $f->{all} = lc join(' ', map { $_->{c} } @{ $f->{L} });
  push @F, $f;
}
close $lf;

# Report units: table rows whole, prose split into sentences.
for my $f (@F) {
  my @U;
  for my $i (0 .. $#{ $f->{L} }) {
    my $l = $f->{L}[$i];
    next if $l->{skip} || $l->{c} eq '';
    if ($l->{row}) {
      push @U, { i => $i, t => $l->{c}, lc => lc $l->{c}, row => 1 } if has_num($l->{c});
    } else {
      for my $s (sentences($l->{c})) {
        push @U, { i => $i, t => $s, lc => lc $s, row => 0 } if has_num($s) || $s =~ $VERB;
      }
    }
  }
  $f->{U} = \@U;
}

# ─── A: negative-scope locator ───────────────────────────────────────────────
my ($nclaims, $npairs, $nmore, @noobj) = (0, 0, 0);
for my $f (@F) {
  for my $i (0 .. $#{ $f->{L} }) {
    my $l = $f->{L}[$i];
    next if $l->{skip} || $l->{row} || $l->{c} eq '';
    for my $s (sentences($l->{c})) {
      next unless $s =~ $NEG;
      $nclaims++;
      my %obj;
      for my $t (keys %{ $l->{it} }) {
        (my $o = $t) =~ s/\\_/_/g;
        next unless $o =~ /^[A-Za-z][\w.\-\/]*$/ && length($o) >= 3 && !$STOP{lc $o};
        $obj{$o} = 1 if index($s, $o) >= 0;
      }
      while ($s =~ /(?<![\w\\-])([A-Za-z][A-Za-z0-9]*(?:[-_.\/][A-Za-z0-9]+)*)/g) {
        my $o = $1;
        next if length($o) < 3;
        next if $o =~ /^[A-Z]+$/;                         # acronyms recur everywhere
        $obj{$o} = 1 if ($o =~ /\d/ && $o =~ /[A-Za-z]{2}/) || $o =~ /[a-z][A-Z]/;
      }
      my @O = sort keys %obj;
      unless (@O) { my $loc = "$f->{rel}:" . ($i + 1); push @noobj, $loc unless @noobj && $noobj[-1] eq $loc; next }

      # Scope terms: the population the statement removes the object from.
      my %S;
      my @scope;
      push @scope, $1 if $s =~ /\bno\s+([\w-]+)\s+claims?\b/i;
      push @scope, $1 if $s =~ /\bnot\s+supported\s+(?:under|for|on|by|in|with)\s+(?:the\s+|this\s+|these\s+)?([\w-]+(?:\s+[\w-]+)?)/i;
      push @scope, $1 if $s =~ /\bunsupported\s+([\w-]+(?:\s+[\w-]+)?)/i;
      push @scope, $1, $2 if $s =~ /(?:^|\s)([\w-]+(?:\s+[\w-]+)?)\s+exclud\w*\s+(?:from\s+)?(?:the\s+)?([\w-]+(?:\s+[\w-]+)?)/i;
      push @scope, $1 if $s =~ /\boutside\s+(?:the|our)\s+scope(?:\s+of)?\s+(?:the\s+)?([\w-]+(?:\s+[\w-]+)?)/i;
      push @scope, $1 if $s =~ /\bnot\s+apply\s+(?:to|under|on|in)\s+(?:the\s+)?([\w-]+(?:\s+[\w-]+)?)/i;
      push @scope, $1 if $s =~ /\bnot\s+evaluated\s+(?:on|under|for|in|with)\s+(?:the\s+)?([\w-]+(?:\s+[\w-]+)?)/i;
      for my $p (@scope) {
        next unless defined $p;
        for my $w (split /[\s-]+/, $p) { my $st = stem($w); $S{$st} = 1 if length($st) >= 4 && !$STOP{$st} && !$obj{$w} }
      }
      my %W;
      for my $w ($s =~ /\b([A-Za-z]{5,})\b/g) { my $st = stem($w); $W{$st} = 1 unless $STOP{$st} || $S{$st} }

      my @cand;
      for my $g (@F) {
        # A scope term in the file name or caption marks a file whose counts are
        # about that population; one only in its text is weaker evidence.
        my $sfile = 0;
        for my $st (keys %S) {
          if ("$g->{base} $g->{caption}" =~ /\b\Q$st\E/) { $sfile += 4 }
          elsif ($g->{all} =~ /\b\Q$st\E/)             { $sfile += 1 }
        }
        my $best;
        for my $u (@{ $g->{U} }) {
          next if $g == $f && $u->{i} == $i;
          for my $o (@O) {
            next unless $u->{lc} =~ /(?<![\w-])\Q${\ lc $o}\E(?![\w])/;
            my $ctx = "$u->{lc} $g->{base} $g->{caption}";
            my $sc = $sfile + 2 * $u->{row};   # pooled counts live in tables
            for my $p (@O) { next if $p eq $o; $sc += 2 if $ctx =~ /(?<![\w-])\Q${\ lc $p}\E(?![\w])/ }
            for my $st (keys %W) { $sc++ if $ctx =~ /\b\Q$st\E/ }
            $best = { sc => $sc, u => $u, o => $o, g => $g } if !$best || $sc > $best->{sc};
          }
        }
        push @cand, $best if $best;
      }
      next unless @cand;
      @cand = sort { $b->{sc} <=> $a->{sc} || $a->{g}{rel} cmp $b->{g}{rel} } @cand;
      my $show = ($TOPN > 0 && @cand > $TOPN) ? $TOPN : scalar @cand;
      for my $c (@cand[0 .. $show - 1]) {
        $npairs++;
        printf "F\t%s:%d: [A] scope statement \"%s\" <-> %s:%d \"%s\" (object: %s) -- consistent, contradiction, or scope too broad?\n",
          $f->{src}, $i + 1, snip($s), $c->{g}{rel}, $c->{u}{i} + 1, snip($c->{u}{t}, 70), $c->{o};
      }
      if (@cand > $show) {
        my @rest = map { "$_->{g}{rel}:" . ($_->{u}{i} + 1) } @cand[$show .. $#cand];
        $nmore += @rest;
        my $list = join(', ', @rest[0 .. ($#rest < 7 ? $#rest : 7)]);
        $list .= sprintf(' (+%d)', @rest - 8) if @rest > 8;
        print "I\t[A] $f->{rel}:" . ($i + 1) . " also named with a number in " . scalar(@rest) . " more file(s): $list\n";
      }
    }
  }
}
print "I\t[A] $nclaims negative-scope statement(s), $npairs pair(s) shown, $nmore lower-ranked report file(s) listed\n";
if (@noobj) {
  my $list = join(', ', @noobj[0 .. ($#noobj < 9 ? $#noobj : 9)]);
  $list .= sprintf(' (+%d)', @noobj - 10) if @noobj > 10;
  print "I\t[A] " . scalar(@noobj) . " statement(s) name no extractable object, check by hand: $list\n";
}

# ─── B: count consistency ────────────────────────────────────────────────────
sub npkey {
  my $np = shift;
  my @k;
  for my $w (split /\s+/, $np) {
    # "the 45 pairs of the test split" names a subset of a larger population;
    # comparing its denominator with the whole is not a drift.
    return '' if $k[0] && $w =~ /^(?:of|in|on|from|per|within)$/i;
    last if $CUT{lc $w};
    push @k, stem($w);
  }
  # One-word heads ("runs", "settings", "configurations") name many different
  # populations in one paper; only a modified noun is specific enough to compare.
  return '' unless @k >= 2 && grep { length($_) >= 3 } @k;
  return '' if $UNIT{ $k[-1] };
  return join(' ', @k);
}

my (%pop, %cell, @kofn);
my $b2 = 0;
my $NP = qr/([A-Za-z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,4})/;
for my $f (@F) {
  for my $i (0 .. $#{ $f->{L} }) {
    my $l = $f->{L}[$i];
    next if $l->{skip} || $l->{c} eq '';
    if ($l->{row}) {
      while ($l->{c} =~ /(?<![\d.,])($NUM)\s*\/\s*($NUM)(?![\d.,])/g) {
        my ($k, $n) = (num($1), num($2));
        push @{ $cell{$n}{$k} }, "$f->{rel}:" . ($i + 1) if $n >= $MIN_DEN;
      }
      next;
    }
    for my $s (sentences($l->{c})) {
      my %seen;
      my $add = sub {
        my ($n, $np) = @_;
        my $key = npkey($np);
        return if $key eq '' || $n < 2 || ($n =~ /^(?:19|20)\d\d$/);
        return if $seen{"$n\t$key"}++;
        push @{ $pop{$key} }, { n => $n, f => $f, i => $i };
      };
      while ($s =~ /(?<![\d.,])($NUM)\s+(?:out\s+)?of\s+(?:the|all(?:\s+the)?|these|those)\s+($NUM)\s+$NP/gi) { $add->(num($2), $3) }
      while ($s =~ /(?<![\d.,])\d+(?:\.\d+)?\s*%\s+of\s+(?:the|all(?:\s+the)?|these)\s+($NUM)\s+$NP/gi) { $add->(num($1), $2) }
      while ($s =~ /\b(?:the|all(?:\s+the)?|these|those)\s+($NUM)\s+$NP/gi) { $add->(num($1), $2) }

      # k of n / k/n spans for B2 and B3.
      my @frac;
      while ($s =~ /(?<![\d.,])($NUM)\s+(?:out\s+)?of\s+(?:the\s+|all\s+)?($NUM)(?![\d.,])/g) { push @frac, [ $-[0], $+[0], num($1), num($2) ] }
      while ($s =~ /(?<![\d.,\/])($NUM)\s*\/\s*($NUM)(?![\d.,\/])/g) { push @frac, [ $-[0], $+[0], num($1), num($2) ] }
      @frac = sort { $a->[0] <=> $b->[0] } @frac;
      my @pct;
      # Only a percentage that opens a parenthesis ("k of n ... (p%") or sits
      # directly before one ("p% (k/n)") restates the fraction. "42 of 45 pairs
      # against a success of 96.7%" is a second quantity, not a restatement.
      while ($s =~ /(\()?\s*(?<![\d.,])(\d+(?:\.(\d+))?)\s*%(\s*\()?/g) { push @pct, [ $-[2], $2, length($3 // ''), $1 ? 1 : 0, $4 ? 1 : 0 ] }
      for my $j (0 .. $#frac) {
        my ($st, $en, $k, $n) = @{ $frac[$j] };
        next if $n == 0 || $k > $n;
        push @kofn, { k => $k, n => $n, f => $f, i => $i, t => $s } if $n >= $MIN_DEN;
        my $next = $j < $#frac ? $frac[$j + 1][0] : length($s);
        my ($p) = grep { $_->[3] && $_->[0] >= $en && $_->[0] < $next && $_->[0] - $en <= 100 } @pct;
        ($p) = grep { $_->[4] && $_->[0] < $st && $st - $_->[0] <= 12 } @pct unless $p;
        next unless $p;
        my ($pv, $d) = ($p->[1], $p->[2]);
        my $v = 100 * $k / $n;
        my $tol = 0.5 * 10**(-$d) + 1e-9;
        # Rounding or truncation at the shown precision; no complement.
        next if abs($v - $pv) <= $tol || ($v - $pv >= 0 && $v - $pv < 10**(-$d));
        $b2++;
        printf "F\t%s:%d: [B2] %s/%s = %.*f%% but the text shows %s%% -- \"%s\"\n",
          $f->{src}, $i + 1, fmt($k), fmt($n), $d + 1, $v, $pv, snip($s);
      }
    }
  }
}

my $b1 = 0;
for my $key (sort keys %pop) {
  my @occ = @{ $pop{$key} };
  my (%byn, %files);
  for my $o (@occ) { push @{ $byn{ $o->{n} } }, $o; $files{ $o->{f}{rel} }{ $o->{n} } = 1 }
  next if keys %byn < 2;
  # Different denominators must sit in different files: within one file,
  # neighbouring subsets ("the 30 settings of R", "the 159 settings") are normal.
  my $cross = 0;
  my @fl = keys %files;
  for my $a (@fl) { for my $bf (@fl) { next if $a eq $bf; for my $n (keys %{ $files{$a} }) { $cross = 1 if grep { $_ != $n } keys %{ $files{$bf} } } } }
  next unless $cross;
  my @ns = sort { @{ $byn{$b} } <=> @{ $byn{$a} } || $a <=> $b } keys %byn;
  my $mode = shift @ns;
  my %u; my @ml = grep { !$u{$_}++ } map { "$_->{f}{rel}:" . ($_->{i} + 1) } @{ $byn{$mode} };
  my $mlist = join(', ', @ml[0 .. ($#ml < 2 ? $#ml : 2)]) . (@ml > 3 ? sprintf(' (+%d)', @ml - 3) : '');
  for my $n (@ns) {
    my $o = $byn{$n}[0];
    $b1++;
    printf "F\t%s:%d: [B1] \"%s\" has denominator %s here but %s in %s\n",
      $o->{f}{src}, $o->{i} + 1, $key, fmt($n), fmt($mode), $mlist;
  }
}

my $b3 = 0;
for my $o (@kofn) {
  next unless $cell{ $o->{n} };
  next if $cell{ $o->{n} }{ $o->{k} };
  my @c = map { my $k = $_; map { fmt($k) . "/" . fmt($o->{n}) . " ($_)" } @{ $cell{ $o->{n} }{$k} } } sort { $a <=> $b } keys %{ $cell{ $o->{n} } };
  $b3++;
  printf "F\t%s:%d: [B3] prose %s of %s, but table cells with this denominator show %s -- \"%s\"\n",
    $o->{f}{src}, $o->{i} + 1, fmt($o->{k}), fmt($o->{n}), join(', ', @c[0 .. ($#c < 2 ? $#c : 2)]), snip($o->{t}, 70);
}
print "I\t[B] $b1 denominator drift(s), $b2 percentage mismatch(es), $b3 prose-vs-table numerator mismatch(es); prose totals vs table column sums are not checked\n";
