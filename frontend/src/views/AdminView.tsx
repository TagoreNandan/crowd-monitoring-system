import React, { useState, useEffect, useMemo } from 'react';
import { RefreshCw, Search, Filter, UserPlus, Trash2, X, Shield } from 'lucide-react';
import type { ActivityLog, AdminOverviewStats, User } from '../types/api';
import {
  getAdminStats,
  getAdminUsers,
  updateAdminUser,
  createAdminUser,
  deleteAdminUser,
  getAdminActivities,
} from '../lib/api';
import { useAuth } from '../context/AuthContext';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';

export const AdminView: React.FC = () => {
  const { token, user: currentUser } = useAuth();
  const [activeTab, setActiveTab] = useState<'activities' | 'users'>('activities');
  const [stats, setStats] = useState<AdminOverviewStats | null>(null);
  const [users, setUsers] = useState<User[]>([]);
  const [activities, setActivities] = useState<ActivityLog[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<string>('ALL');
  const [selectedSeverity, setSelectedSeverity] = useState<string>('ALL');
  const [userSearch, setUserSearch] = useState<string>('');
  const [activitySearch, setActivitySearch] = useState<string>('');

  // Add User Modal State
  const [isAddUserOpen, setIsAddUserOpen] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [newEmail, setNewEmail] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newFullName, setNewFullName] = useState('');
  const [newRole, setNewRole] = useState<'analyst' | 'admin'>('analyst');
  const [addUserLoading, setAddUserLoading] = useState(false);
  const [addUserError, setAddUserError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      const [statsData, usersData, activitiesData] = await Promise.all([
        getAdminStats(token || undefined).catch(() => null),
        getAdminUsers(token || undefined).catch(() => []),
        getAdminActivities(100, token || undefined).catch(() => []),
      ]);

      if (statsData) setStats(statsData);
      setUsers(usersData);
      setActivities(activitiesData);
    } catch (e) {
      console.error('Failed to load admin telemetry', e);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 8000);
    return () => clearInterval(interval);
  }, [token]);

  const handleRefresh = () => {
    setRefreshing(true);
    loadData();
  };

  const handleToggleUserStatus = async (user: User) => {
    if (!token) return;
    try {
      await updateAdminUser(user.id, { is_active: !user.is_active }, token);
      await loadData();
    } catch (err: any) {
      alert(err.message || 'Failed to update user status');
    }
  };

  const handleToggleRole = async (user: User) => {
    if (!token) return;
    const newRole = user.role === 'admin' ? 'analyst' : 'admin';
    try {
      await updateAdminUser(user.id, { role: newRole }, token);
      await loadData();
    } catch (err: any) {
      alert(err.message || 'Failed to change role');
    }
  };

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) return;
    setAddUserLoading(true);
    setAddUserError(null);
    try {
      await createAdminUser(
        {
          username: newUsername.trim(),
          email: newEmail.trim(),
          password: newPassword,
          full_name: newFullName.trim() || undefined,
          role: newRole,
        },
        token
      );
      setIsAddUserOpen(false);
      setNewUsername('');
      setNewEmail('');
      setNewPassword('');
      setNewFullName('');
      setNewRole('analyst');
      await loadData();
    } catch (err: any) {
      setAddUserError(err.message || 'Failed to create user');
    } finally {
      setAddUserLoading(false);
    }
  };

  const handleDeleteUser = async (user: User) => {
    if (!token) return;
    if (!confirm(`Are you sure you want to permanently delete user "${user.username}"?`)) return;
    try {
      await deleteAdminUser(user.id, token);
      await loadData();
    } catch (err: any) {
      alert(err.message || 'Failed to delete user');
    }
  };

  // Dynamically extract categories and severities from real activity records
  const dynamicCategories = useMemo(() => {
    const cats = new Set(activities.map((a) => a.category).filter(Boolean));
    return ['ALL', ...Array.from(cats)];
  }, [activities]);

  const dynamicSeverities = useMemo(() => {
    const sevs = new Set(activities.map((a) => a.severity).filter(Boolean));
    return ['ALL', ...Array.from(sevs)];
  }, [activities]);

  // Dynamically computed metrics
  const totalUsers = stats?.total_users ?? users.length;
  const adminCount = stats?.admin_users ?? users.filter((u) => u.role === 'admin').length;
  const activeCount = stats?.active_users ?? users.filter((u) => u.is_active).length;
  const totalActivities = stats?.total_activities ?? activities.length;
  const dbStatus = stats?.database_status ?? 'ONLINE';
  const dbEngine = stats?.database_engine ? `${stats.database_engine.toUpperCase()} active` : 'Active';

  const filteredActivities = activities.filter((act) => {
    const matchesCategory = selectedCategory === 'ALL' || act.category === selectedCategory;
    const matchesSeverity = selectedSeverity === 'ALL' || act.severity === selectedSeverity;
    const q = activitySearch.toLowerCase();
    const matchesSearch =
      !q ||
      act.action.toLowerCase().includes(q) ||
      act.username.toLowerCase().includes(q) ||
      (act.details && act.details.toLowerCase().includes(q));
    return matchesCategory && matchesSeverity && matchesSearch;
  });

  const filteredUsers = users.filter((u) => {
    const q = userSearch.toLowerCase();
    return (
      u.username.toLowerCase().includes(q) ||
      u.email.toLowerCase().includes(q) ||
      (u.full_name && u.full_name.toLowerCase().includes(q))
    );
  });

  const formatTimestamp = (ts: string | null | undefined): string => {
    if (!ts) return 'Never';
    // Ensure UTC timestamp has timezone indicator so JavaScript converts to local OS time
    const iso = ts.endsWith('Z') || ts.includes('+') ? ts : `${ts}Z`;
    const d = new Date(iso);
    if (isNaN(d.getTime())) {
      const fallback = new Date(ts);
      return isNaN(fallback.getTime()) ? String(ts) : fallback.toLocaleString();
    }

    return d.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
      hour12: true,
    });
  };

  const renderSeverityBadge = (severity: string) => {
    switch (severity?.toUpperCase()) {
      case 'CRITICAL':
        return <Badge variant="critical">CRITICAL</Badge>;
      case 'WARNING':
        return <Badge variant="warning">WARNING</Badge>;
      case 'INFO':
        return <Badge variant="safe">INFO</Badge>;
      default:
        return <Badge variant="outline">{severity || 'INFO'}</Badge>;
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8 animate-in fade-in duration-150">
      {/* Header section */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-zinc-800">
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-zinc-100">Administrator Console</h1>
            <Badge variant="outline">Admin Access Active</Badge>
          </div>
          <p className="text-sm text-zinc-400">
            System audit telemetry, operational logs, and user account governance.
          </p>
        </div>

        <Button
          variant="outline"
          size="sm"
          onClick={handleRefresh}
          disabled={refreshing}
          className="self-start sm:self-center"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin text-zinc-300' : 'text-zinc-500'}`} />
          <span>Refresh</span>
        </Button>
      </div>

      {/* Dynamic KPI Stats Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800">
          <span className="text-xs text-zinc-500 block font-mono">Total Accounts</span>
          <span className="text-2xl font-semibold font-mono text-zinc-100">{totalUsers}</span>
          <span className="text-[11px] text-zinc-500 block mt-1 font-mono">
            {adminCount} Admin / {activeCount} Active
          </span>
        </div>

        <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800">
          <span className="text-xs text-zinc-500 block font-mono">Total Activities</span>
          <span className="text-2xl font-semibold font-mono text-zinc-100">{totalActivities}</span>
          <span className="text-[11px] text-zinc-500 block mt-1">Recorded audit entries</span>
        </div>

        <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800">
          <span className="text-xs text-zinc-500 block font-mono">Database State</span>
          <span className="text-2xl font-semibold font-mono text-emerald-400">{dbStatus}</span>
          <span className="text-[11px] text-zinc-500 block mt-1 font-mono">{dbEngine}</span>
        </div>

        <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800">
          <span className="text-xs text-zinc-500 block font-mono">Active Operator</span>
          <span className="text-2xl font-semibold font-mono text-zinc-100 truncate block">
            {currentUser?.username ?? '—'}
          </span>
          <span className="text-[11px] text-zinc-500 font-mono block mt-1 truncate">
            {currentUser?.email ?? '—'}
          </span>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-zinc-800 space-x-6">
        <button
          onClick={() => setActiveTab('activities')}
          className={`pb-3 text-xs font-mono uppercase tracking-wider transition-colors border-b-2 cursor-pointer ${
            activeTab === 'activities'
              ? 'border-zinc-100 text-zinc-100 font-semibold'
              : 'border-transparent text-zinc-400 hover:text-zinc-200'
          }`}
        >
          Activity Log ({activities.length})
        </button>

        <button
          onClick={() => setActiveTab('users')}
          className={`pb-3 text-xs font-mono uppercase tracking-wider transition-colors border-b-2 cursor-pointer ${
            activeTab === 'users'
              ? 'border-zinc-100 text-zinc-100 font-semibold'
              : 'border-transparent text-zinc-400 hover:text-zinc-200'
          }`}
        >
          User Accounts ({users.length})
        </button>
      </div>

      {/* TAB 1: ACTIVITIES */}
      {activeTab === 'activities' && (
        <div className="space-y-4">
          {/* Dynamic Filter Bar */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3 bg-zinc-900/30 border border-zinc-800 rounded-lg">
            <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-400 font-mono">
              <Filter className="w-3.5 h-3.5 text-zinc-500 mr-1" />
              <span>Category:</span>
              {dynamicCategories.map((cat) => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-2 py-0.5 rounded text-xs transition-colors cursor-pointer ${
                    selectedCategory === cat
                      ? 'bg-zinc-800 text-zinc-100 font-medium'
                      : 'text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  {cat}
                </button>
              ))}

              {dynamicSeverities.length > 2 && (
                <>
                  <span className="ml-2 pl-2 border-l border-zinc-800">Severity:</span>
                  {dynamicSeverities.map((sev) => (
                    <button
                      key={sev}
                      onClick={() => setSelectedSeverity(sev)}
                      className={`px-2 py-0.5 rounded text-xs transition-colors cursor-pointer ${
                        selectedSeverity === sev
                          ? 'bg-zinc-800 text-zinc-100 font-medium'
                          : 'text-zinc-500 hover:text-zinc-300'
                      }`}
                    >
                      {sev}
                    </button>
                  ))}
                </>
              )}
            </div>

            <div className="relative max-w-xs w-full sm:w-48">
              <input
                type="text"
                value={activitySearch}
                onChange={(e) => setActivitySearch(e.target.value)}
                placeholder="Search events..."
                className="w-full pl-7 pr-3 py-1 bg-zinc-900/60 border border-zinc-800 rounded-md text-xs text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-400"
              />
              <Search className="w-3 h-3 text-zinc-500 absolute left-2 top-2" />
            </div>
          </div>

          {/* Activity Table */}
          <div className="rounded-lg bg-zinc-900/30 border border-zinc-800 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-zinc-300">
                <thead className="bg-zinc-900/60 text-zinc-400 uppercase tracking-wider font-mono text-[11px] border-b border-zinc-800">
                  <tr>
                    <th className="py-2.5 px-4 font-medium">Timestamp</th>
                    <th className="py-2.5 px-4 font-medium">User</th>
                    <th className="py-2.5 px-4 font-medium">Action</th>
                    <th className="py-2.5 px-4 font-medium">Category</th>
                    <th className="py-2.5 px-4 font-medium">Details</th>
                    <th className="py-2.5 px-4 font-medium">Severity</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/50 font-mono text-xs">
                  {filteredActivities.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="py-8 text-center text-zinc-500">
                        No activity records matching query.
                      </td>
                    </tr>
                  ) : (
                    filteredActivities.map((act) => (
                      <tr key={act.id} className="hover:bg-zinc-800/20 transition-colors">
                        <td className="py-2.5 px-4 text-zinc-400 whitespace-nowrap">
                          {formatTimestamp(act.timestamp)}
                        </td>
                        <td className="py-2.5 px-4 text-zinc-200 font-medium">{act.username}</td>
                        <td className="py-2.5 px-4 text-zinc-100">{act.action}</td>
                        <td className="py-2.5 px-4 text-zinc-400">{act.category}</td>
                        <td className="py-2.5 px-4 max-w-xs truncate text-zinc-300 font-sans">
                          {act.details || '—'}
                        </td>
                        <td className="py-2.5 px-4 whitespace-nowrap">{renderSeverityBadge(act.severity)}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: USER MANAGEMENT */}
      {activeTab === 'users' && (
        <div className="space-y-4">
          {/* User Search & Add User Bar */}
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 p-3 bg-zinc-900/30 border border-zinc-800 rounded-lg">
            <div className="relative flex-1 max-w-sm">
              <input
                type="text"
                value={userSearch}
                onChange={(e) => setUserSearch(e.target.value)}
                placeholder="Search users..."
                className="w-full pl-8 pr-3 py-1 bg-zinc-900/60 border border-zinc-800 rounded-md text-xs text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-400"
              />
              <Search className="w-3.5 h-3.5 text-zinc-500 absolute left-2.5 top-2" />
            </div>

            <div className="flex items-center gap-3 self-end sm:self-auto">
              <span className="text-[11px] font-mono text-zinc-500">{filteredUsers.length} accounts</span>
              <Button
                variant="default"
                size="sm"
                onClick={() => {
                  setAddUserError(null);
                  setIsAddUserOpen(true);
                }}
                className="text-xs h-7 px-2.5 flex items-center gap-1.5"
              >
                <UserPlus className="w-3.5 h-3.5" />
                <span>Add User</span>
              </Button>
            </div>
          </div>

          {/* User Table */}
          <div className="rounded-lg bg-zinc-900/30 border border-zinc-800 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-zinc-300">
                <thead className="bg-zinc-900/60 text-zinc-400 uppercase tracking-wider font-mono text-[11px] border-b border-zinc-800">
                  <tr>
                    <th className="py-2.5 px-4 font-medium">Username</th>
                    <th className="py-2.5 px-4 font-medium">Email</th>
                    <th className="py-2.5 px-4 font-medium">Role</th>
                    <th className="py-2.5 px-4 font-medium">Status</th>
                    <th className="py-2.5 px-4 font-medium">Last Login</th>
                    <th className="py-2.5 px-4 font-medium text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/50 text-xs">
                  {filteredUsers.map((u) => {
                    const isSelf = u.id === currentUser?.id;
                    return (
                      <tr key={u.id} className="hover:bg-zinc-800/20 transition-colors">
                        <td className="py-2.5 px-4">
                          <span className="font-medium text-zinc-200">{u.username}</span>
                          {isSelf && <span className="ml-1.5 text-[10px] text-zinc-500">(You)</span>}
                        </td>
                        <td className="py-2.5 px-4 font-mono text-zinc-400">{u.email}</td>
                        <td className="py-2.5 px-4">
                          <Badge variant={u.role === 'admin' ? 'outline' : 'secondary'}>
                            {u.role.toUpperCase()}
                          </Badge>
                        </td>
                        <td className="py-2.5 px-4">
                          <Badge variant={u.is_active ? 'safe' : 'critical'}>
                            {u.is_active ? 'Active' : 'Disabled'}
                          </Badge>
                        </td>
                        <td className="py-2.5 px-4 font-mono text-zinc-400">
                          {formatTimestamp(u.last_login)}
                        </td>
                        <td className="py-2.5 px-4 text-right space-x-1.5">
                          {!isSelf && (
                            <>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleToggleRole(u)}
                                className="text-[11px] h-7 px-2"
                              >
                                {u.role === 'admin' ? 'Demote' : 'Promote'}
                              </Button>
                              <Button
                                variant={u.is_active ? 'secondary' : 'default'}
                                size="sm"
                                onClick={() => handleToggleUserStatus(u)}
                                className="text-[11px] h-7 px-2"
                              >
                                {u.is_active ? 'Deactivate' : 'Activate'}
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleDeleteUser(u)}
                                className="text-[11px] h-7 px-2 text-rose-400 hover:text-rose-300 hover:bg-rose-950/40"
                                title="Delete user account"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </Button>
                            </>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Modal: Add User */}
      {isAddUserOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs animate-in fade-in duration-150">
          <div className="relative w-full max-w-sm bg-zinc-950 border border-zinc-800 rounded-lg shadow-2xl p-6 text-zinc-100">
            <button
              onClick={() => setIsAddUserOpen(false)}
              className="absolute top-4 right-4 p-1 rounded-md text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 transition-colors cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="mb-5 space-y-1">
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-md bg-zinc-900 border border-zinc-800 text-zinc-300">
                  <UserPlus className="w-4 h-4" />
                </div>
                <h2 className="text-base font-semibold tracking-tight text-zinc-100">
                  Add New User
                </h2>
              </div>
              <p className="text-xs text-zinc-400">
                Provision a new operator or analyst account.
              </p>
            </div>

            {addUserError && (
              <div className="mb-4 p-2.5 rounded-md bg-rose-950/40 border border-rose-800/50 text-rose-300 text-xs">
                {addUserError}
              </div>
            )}

            <form onSubmit={handleCreateUser} className="space-y-3">
              <div className="space-y-1">
                <label className="block text-xs font-medium text-zinc-300">Username *</label>
                <input
                  type="text"
                  required
                  value={newUsername}
                  onChange={(e) => setNewUsername(e.target.value)}
                  placeholder="e.g. jdoe"
                  className="w-full px-3 py-1.5 bg-zinc-900/60 border border-zinc-800 rounded-md text-xs text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-400"
                />
              </div>

              <div className="space-y-1">
                <label className="block text-xs font-medium text-zinc-300">Email *</label>
                <input
                  type="email"
                  required
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  placeholder="e.g. jdoe@example.com"
                  className="w-full px-3 py-1.5 bg-zinc-900/60 border border-zinc-800 rounded-md text-xs text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-400"
                />
              </div>

              <div className="space-y-1">
                <label className="block text-xs font-medium text-zinc-300">Full Name</label>
                <input
                  type="text"
                  value={newFullName}
                  onChange={(e) => setNewFullName(e.target.value)}
                  placeholder="e.g. John Doe"
                  className="w-full px-3 py-1.5 bg-zinc-900/60 border border-zinc-800 rounded-md text-xs text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-400"
                />
              </div>

              <div className="space-y-1">
                <label className="block text-xs font-medium text-zinc-300">Password *</label>
                <input
                  type="password"
                  required
                  minLength={6}
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="At least 6 characters"
                  className="w-full px-3 py-1.5 bg-zinc-900/60 border border-zinc-800 rounded-md text-xs text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-400"
                />
              </div>

              <div className="space-y-1">
                <label className="block text-xs font-medium text-zinc-300">Role</label>
                <select
                  value={newRole}
                  onChange={(e) => setNewRole(e.target.value as 'analyst' | 'admin')}
                  className="w-full px-3 py-1.5 bg-zinc-900 border border-zinc-800 rounded-md text-xs text-zinc-100 focus:outline-none focus:ring-1 focus:ring-zinc-400"
                >
                  <option value="analyst">Analyst (Standard User)</option>
                  <option value="admin">Administrator (Full Access)</option>
                </select>
              </div>

              <div className="pt-2 flex gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setIsAddUserOpen(false)}
                  className="w-1/2"
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  variant="default"
                  size="sm"
                  isLoading={addUserLoading}
                  className="w-1/2"
                >
                  Create User
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
